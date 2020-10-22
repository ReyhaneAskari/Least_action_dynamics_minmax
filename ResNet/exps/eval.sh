#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
python test.py \
--img_size 32 \
--model sngan_cifar10 \
--latent_dim 128 \
--gf_dim 256 \
--g_spectral_norm False \
--load_path best_slead/checkpoint_best_1.pth \
--exp_name test_best_slead \
--num_eval_imgs 5000