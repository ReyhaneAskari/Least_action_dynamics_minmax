# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import models
from utils.utils import set_log_dir, create_logger
from utils.inception_score import _init_inception
from utils.inception_score import get_inception_score
from utils.inception_score import create_inception_graph
from utils.inception_score import check_or_download_inception
from utils.fid_score_pytorch import calculate_fid

import torch.nn as nn

import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from itertools import chain
import datasets
from pathlib import Path
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def validate(args, fixed_z, gen_net: nn.Module, writer_dict):
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    gen_net = gen_net.eval()
    global_steps = writer_dict['valid_global_steps']
    eval_iter = args.num_eval_imgs // args.eval_batch_size
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir)
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # compute IS
    inception_score, std = get_inception_score(img_list)
    print('------------------------ Inception Score ------------------------')
    print(inception_score)

    print('------------------------ FID pytorch ------------------------')
    print(fid_score)

    # Generate a batch of images
    sample_dir = os.path.join(args.path_helper['sample_path'], 'sample_dir')
    Path(sample_dir).mkdir(exist_ok=True)

    sample_imgs = gen_net(fixed_z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0)
    img_grid = make_grid(sample_imgs, nrow=5).to('cpu', torch.uint8).numpy()
    file_name = os.path.join(sample_dir, f'final_fid_{fid}_inception_score{inception_score}.png')
    imsave(file_name, img_grid.swapaxes(0, 1).swapaxes(1, 2))

    writer_dict['valid_global_steps'] = global_steps + 1
    return inception_score, fid


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert args.load_path.endswith('.pth')
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval('models.' + args.model + '.Generator')(args=args).cuda()

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))

    # set writer
    logger.info(f'=> resuming from {args.load_path}')
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    if 'avg_gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    else:
        gen_net.load_state_dict(checkpoint)
        logger.info(f'=> loaded checkpoint {checkpoint_file}')

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': 0,
    }
    inception_score, fid_score = validate(args, fixed_z, gen_net, writer_dict)
    logger.info(f'Inception score: {inception_score}, FID score: {fid_score}.')


if __name__ == '__main__':
    main()
