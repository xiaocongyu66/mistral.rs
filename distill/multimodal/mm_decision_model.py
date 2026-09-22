"""Multimodal decision model: Apeireth-Decis backbone + CLIP visual encoder.

Architecture:
  image → CLIP-ViT-B-16 → proj layer → visual tokens (prefix)
  text state + question + candidates → Apeireth backbone → decision head
  SWD-H loss aligns hidden states between jev-text-teacher and visual student.

Training signal: jev labels the TEXT description of the image (two-stage bridge),
the student learns to produce the same distribution from the IMAGE directly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEncoder(nn.Module):
    """CLIP-ViT-B-16 wrapper: image → projected visual token prefix."""

    def __init__(self, backbone_dim: int, freeze_clip: bool = True):
        super().__init__()
        from transformers import CLIPVisionModel
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        clip_dim = self.clip.config.hidden_size  # 768
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
        # projection: CLIP 768 → backbone hidden (e.g. 1024 for Qwen3-0.6B)
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
            nn.Linear(backbone_dim, backbone_dim),
        )
        # learnable modality gate (start near 0 so text-only behaviour is preserved)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, 224, 224] → [B, num_patches, backbone_dim]"""
        out = self.clip.vision_model(pixel_values=pixel_values)
        # use CLS + patch embeddings: [B, 197, 768]
        feats = out.last_hidden_state
        projected = self.proj(feats)  # [B, 197, backbone_dim]
        return projected * torch.sigmoid(self.gate)


class SWDProjector(nn.Module):
    """From Light-MER: teacher→student projection with orthogonal frozen init."""

    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.teacher_proj = nn.Linear(teacher_dim, student_dim, bias=False)
        nn.init.orthogonal_(self.teacher_proj.weight)
        for p in self.teacher_proj.parameters():
            p.requires_grad = False
        self.student_proj = nn.Sequential(
            nn.Linear(student_dim, student_dim, bias=False),
            nn.LayerNorm(student_dim),
        )
        nn.init.eye_(self.student_proj[0].weight)

    def forward(self, teacher_feats, student_feats):
        return self.teacher_proj(teacher_feats), self.student_proj(student_feats)


def sliced_wasserstein_loss(teacher_feats, student_feats, n_projections=100, p=2):
    """From Light-MER ot_loss.py:163."""
    batch_size, seq_len, dim = teacher_feats.shape
    device = teacher_feats.device
    directions = F.normalize(torch.randn(n_projections, dim, device=device), dim=-1)
    t_proj = torch.einsum("bsd,pd->bsp", teacher_feats, directions)
    s_proj = torch.einsum("bsd,pd->bsp", student_feats, directions)
    total = torch.tensor(0.0, device=device, dtype=torch.float32)
    count = 0
    for b in range(batch_size):
        t_sorted, _ = t_proj[b].sort(dim=0)
        s_sorted, _ = s_proj[b].sort(dim=0)
        if p == 1:
            total += torch.abs(t_sorted - s_sorted).mean()
        else:
            total += ((t_sorted - s_sorted) ** 2).mean().sqrt()
        count += 1
    return total / max(count, 1)


class MultimodalDecisionModel(nn.Module):
    """Wraps the text-only DecisionModel with a visual prefix encoder."""

    def __init__(self, text_model, backbone_dim: int, teacher_dim: int = 4096):
        super().__init__()
        self.text_model = text_model  # the existing DecisionModel
        self.visual = VisualEncoder(backbone_dim)
        self.swd_proj = SWDProjector(teacher_dim, backbone_dim)

    def forward(self, input_ids, attention_mask, pixel_values=None,
                teacher_hidden=None):
        """Returns (logits, loss_dict).

        If pixel_values is provided, visual tokens are prepended to the
        input embeddings before the backbone forward.
        If teacher_hidden is provided, SWD-H loss is computed.
        """
        losses = {}
        if pixel_values is not None:
            visual_tokens = self.visual(pixel_values)  # [B, 197, dim]
            losses["visual_gate"] = self.visual.gate.item()

        # delegate to text_model, injecting visual prefix if available
        if pixel_values is not None and hasattr(self.text_model, "backbone"):
            # get input embeddings
            emb_layer = self.text_model.backbone.get_input_embeddings()
            inputs_embeds = emb_layer(input_ids)  # [B, seq, dim]
            # prepend visual tokens
            full_embeds = torch.cat([visual_tokens, inputs_embeds], dim=1)
            vis_mask = torch.ones(visual_tokens.shape[:2], device=input_ids.device,
                                  dtype=attention_mask.dtype)
            full_mask = torch.cat([vis_mask, attention_mask], dim=1)
            logits, _ = self.text_model(
                input_ids=None, inputs_embeds=full_embeds,
                attention_mask=full_mask)
        else:
            logits, _ = self.text_model(input_ids=input_ids,
                                         attention_mask=attention_mask)

        if teacher_hidden is not None:
            proj_t, proj_s = self.swd_proj(teacher_hidden.float(),
                                            logits.detach().float())
            losses["swd"] = sliced_wasserstein_loss(proj_t.unsqueeze(0),
                                                     proj_s.unsqueeze(0))

        return logits, losses
