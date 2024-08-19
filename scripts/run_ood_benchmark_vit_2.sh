#!/bin/bash
ctx_init=a_photo_of_a
CUDA_VISIBLE_DEVICES=0 python tda_runner_2.py   --config configs \
                                                --wandb-log \
                                                --datasets I/A/V/R/K \
                                                --data-root '/root/autodl-tmp/dataset/tta_data' \
                                                --backbone ViT-B/16 \
                                                --tpt \
                                                --ctx_init ${ctx_init}
                                                # I/A/V/R/K  --wandb-log \