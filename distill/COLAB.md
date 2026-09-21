# Colab 训练运行手册（任务 #13）

目标：在免费 T4 上用我们的 17760 题工单数据微调 NanoJev 架构（warm start 自
C-Tianyu/NanoJev 发布 checkpoint），产出工单域学生模型 + 校准温度。

## 步骤（Colab 里逐格执行）

```bash
# 1. 环境（T4 免费档即可，0.6B FP32 + BF16 autocast 显存 ~8GB）
!git clone https://github.com/TianyuCodings/NanoJev.git /content/NanoJev
!pip -q install transformers safetensors huggingface_hub

# 2. 取我们的训练数据（decide 分支）
!curl -sL -o /content/unified.jsonl https://raw.githubusercontent.com/xiaocongyu66/mistral.rs/decide/distill/nanojev_format/unified.jsonl
!curl -sL -o /content/colab_train.py   https://raw.githubusercontent.com/xiaocongyu66/mistral.rs/decide/distill/colab_train.py

# 3. 取 NanoJev 发布 checkpoint（warm start，2.27GB）
from huggingface_hub import snapshot_download
p = snapshot_download("C-Tianyu/NanoJev", revision="unified-games-v1",
                      local_dir="/content/nanojev-ckpt",
                      allow_patterns=["best.safetensors"])
```

```python
# 4. 训练（约 20-40 分钟，取决于 steps）
!python /content/colab_train.py \
  --nanojev-scripts /content/NanoJev/scripts \
  --input /content/unified.jsonl \
  --output-dir /content/run1 \
  --init-checkpoint /content/nanojev-ckpt/best.safetensors \
  --steps 600 --head-steps 24 --eval-every 50 --skip-native-baseline
```

## 产物与验收

- `run1/summary.json`：dev teacher CE（对比它的游戏域 dev CE 0.48 数量级即可，
  任务不同没有绝对可比性）、test/ood 指标、显存峰值
- `run1/temperature.json`：校准温度 T（替代默认 3.0，这是 NanoJev 契约的
  calibration split 用法）
- `run1/best.safetensors`：工单域 DecisionModel 权重 → 阶段2改（candle 决策
  引擎）的加载目标

## 对照实验（可选）

- `--init-checkpoint` 不传 = 从底模全新训练，与 warm start 对比看迁移收益
- `--objective gold` = 硬目标训练，与 teacher 软目标对比看蒸馏收益
