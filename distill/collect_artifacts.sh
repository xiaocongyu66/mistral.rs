#!/usr/bin/env bash
# 从 Colab 会话取回训练产物：小文件入 git，权重传 GitHub release
# 用法: SESSION=a5eb9a RUN=/content/run1 TAG=support-triage-v1 ./distill/collect_artifacts.sh
set -euo pipefail
SESSION="${SESSION:?need SESSION}"
RUN="${RUN:-/content/run1}"
TAG="${TAG:?need TAG}"
REPO="${REPO:-xiaocongyu66/nanojev-mirror}"
DEST="${DEST:-/root/trainrun}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$DEST"
for f in summary.json temperature.json config.json train_log.json \
         timing.json target_audit.json predictions.jsonl; do
  colab download "$RUN/$f" "$DEST/$f" -s "$SESSION" 2>/dev/null || echo "skip $f"
done
colab download "$RUN/best.safetensors" "$DEST/best.safetensors" -s "$SESSION"

echo "--- summary ---"; cat "$DEST/summary.json" 2>/dev/null || true
echo "--- temperature ---"; cat "$DEST/temperature.json" 2>/dev/null || true

gh release create "$TAG" --repo "$REPO" \
  --title "Fine-tuned student ($TAG)" \
  --notes "Trained on 17.7k teacher-labeled support-triage questions, warm start from unified-games-v1. See checksums and metrics in assets."
split -b 1900m -d --additional-suffix=.part "$DEST/best.safetensors" /tmp/student.part-
sha256sum /tmp/student.part-* "$DEST/best.safetensors" > "$DEST/checksums.txt"
sed -i "s|  /tmp/student.part-|  student.safetensors.part-|; s|  $DEST/best.safetensors|  student.safetensors|" "$DEST/checksums.txt"
gh release upload "$TAG" --repo "$REPO" /tmp/student.part-* "$DEST/checksums.txt" "$DEST"/summary.json "$DEST"/temperature.json "$DEST"/config.json
echo "done -> https://github.com/$REPO/releases/tag/$TAG"
