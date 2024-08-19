#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python tda_runner.py     --config configs \
                                                --wandb-log \
                                                --datasets I/A/V/R/K \
                                                --data-root '/root/autodl-tmp/dataset/tta_data' \
                                                --backbone ViT-B/16
                                                # I/A/V/R/K  --wandb-log \