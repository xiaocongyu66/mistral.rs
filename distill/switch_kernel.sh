#!/bin/bash
# Switch between GPU and TPU kernel for Kaggle
# Usage: ./switch_kernel.sh gpu|tpu
cd /root/kaggle-train
if [ "$1" = "tpu" ]; then
    cp /root/mistral.rs/distill/kernel-metadata-tpu.json kernel-metadata.json
    echo "Switched to TPU (duan-moe-v9-tpu)"
else
    cp /root/mistral.rs/distill/kernel-metadata-gpu.json kernel-metadata.json
    echo "Switched to GPU (duan-moe-v9)"
fi
