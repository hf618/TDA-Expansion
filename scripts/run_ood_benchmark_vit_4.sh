#!/bin/bash
ctx_init=a_photo_of_a
n_ctx=4
CUDA_VISIBLE_DEVICES=1 python tda_runner_4.py   --config configs \
                                                --datasets I/A/V/R/K \
                                                --data-root '/root/autodl-tmp/dataset/tta_data' \
                                                --backbone ViT-B/16 \
                                                --tpt \
                                                --ctx_init ${ctx_init} \
                                                --n_ctx ${n_ctx} \
                                                --wandb-log \
                                                --print_freq 1000
#caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101 --wandb-log \
# I/A/V/R/K