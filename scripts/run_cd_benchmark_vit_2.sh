#!/bin/bash
ctx_init=a_photo_of_a
CUDA_VISIBLE_DEVICES=0 python tda_runner_2.py   --config configs \
                                                --wandb-log \
                                                --datasets caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101 \
                                                --data-root '/root/autodl-tmp/dataset/tta_data' \
                                                --backbone ViT-B/16 \
                                                --tpt \
                                                --ctx_init ${ctx_init}
#caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101