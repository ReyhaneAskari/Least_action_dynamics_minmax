# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
from itertools import chain

from utils.inception_score import get_inception_score
from torch.nn.utils import parameters_to_vector
from utils.optim import parameters_grad_to_vector
from utils.fid_score_pytorch import calculate_fid
from pathlib import Path


logger = logging.getLogger(__name__)


def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    dis_params_flatten = parameters_to_vector(dis_net.parameters())
    gen_params_flatten = parameters_to_vector(gen_net.parameters())
    bce_loss = nn.BCEWithLogitsLoss()

    if args.optimizer == 'sLead_Adam':
        # just to fill-up the grad buffers
        imgs = iter(train_loader).__next__()[0]
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        fake_imgs = gen_net(z)
        fake_validity = dis_net(fake_imgs)
        d_loss = torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        g_loss = -torch.mean(fake_validity)
        (0.0 * d_loss).backward(create_graph=True)
        (0.0 * g_loss).backward(create_graph=True)

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z)
        assert fake_imgs.size() == real_imgs.size()
        fake_validity = dis_net(fake_imgs)

        # cal loss
        if args.loss_type == 'hinge':
            d_loss = torch.mean(
                nn.ReLU(inplace=True)(1.0 - real_validity)) + torch.mean(
                nn.ReLU(inplace=True)(1 + fake_validity))
        elif args.loss_type == 'bce':
            fake_labels = torch.zeros(imgs.shape[0]).cuda()
            real_labels = torch.ones(imgs.shape[0]).cuda()
            real_loss = bce_loss(real_validity.squeeze(), real_labels)
            fake_loss = bce_loss(fake_validity.squeeze(), fake_labels)
            d_loss = real_loss + fake_loss

        if args.optimizer == 'Adam':
            dis_optimizer.zero_grad()
            d_loss.backward()
            dis_optimizer.step()
        elif args.optimizer == 'sLead_Adam':
            # if global_steps % args.n_critic == 0:
                gradsD = torch.autograd.grad(
                    outputs=d_loss, inputs=(dis_net.parameters()),
                    create_graph=True)
                for p, g in zip(dis_net.parameters(), gradsD):
                    p.grad = g
                gen_params_flatten_prev = gen_params_flatten + 0.0
                gen_params_flatten = parameters_to_vector(gen_net.parameters()) + 0.0
                grad_gen_params_flatten = parameters_grad_to_vector(gen_net.parameters())
                delta_gen_params_flatten = gen_params_flatten - gen_params_flatten_prev
                vjp_dis = torch.autograd.grad(
                    grad_gen_params_flatten, dis_net.parameters(),
                    grad_outputs=delta_gen_params_flatten, retain_graph=True)
                dis_optimizer.step(vjps=vjp_dis)
            # else:
            #     # do regular adam
            #     dis_optimizer.zero_grad()
            #     d_loss.backward()
            #     dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            # cal loss
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)
            if args.loss_type == 'hinge':
                g_loss = -torch.mean(fake_validity)
            elif args.loss_type == 'bce':
                real_labels = torch.ones(args.gen_batch_size).cuda()
                g_loss = bce_loss(fake_validity.squeeze(), real_labels)

            if args.optimizer == 'Adam':
                gen_optimizer.zero_grad()
                g_loss.backward()
                gen_optimizer.step()

            elif args.optimizer == 'sLead_Adam':
                gradsG = torch.autograd.grad(
                    outputs=g_loss, inputs=(gen_net.parameters()),
                    create_graph=True)
                for p, g in zip(gen_net.parameters(), gradsG):
                    p.grad = g

                dis_params_flatten_prev = dis_params_flatten + 0.0
                dis_params_flatten = parameters_to_vector(dis_net.parameters()) + 0.0
                grad_dis_params_flatten = parameters_grad_to_vector(dis_net.parameters())
                delta_dis_params_flatten = dis_params_flatten - dis_params_flatten_prev
                vjp_gen = torch.autograd.grad(
                    grad_dis_params_flatten, gen_net.parameters(),
                    grad_outputs=delta_dis_params_flatten)

                gen_optimizer.step(vjps=vjp_gen)

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1


def validate(args, fixed_z, gen_net: nn.Module, writer_dict, train_loader, epoch):
    gen_net = gen_net.eval()
    global_steps = writer_dict['valid_global_steps']
    gen_net = gen_net.eval()
    eval_iter = args.num_eval_imgs // args.eval_batch_size

    # skip IS
    inception_score = 0

    # compute FID
    sample_list = []
    for i in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
        samples = gen_net(z)
        sample_list.append(samples.data.cpu().numpy())

    new_sample_list = list(chain.from_iterable(sample_list))
    fake_image_np = np.concatenate([img[None] for img in new_sample_list], 0)

    real_image_np = []
    for i, (images, _) in enumerate(train_loader):
        real_image_np += [images.data.numpy()]
        batch_size = real_image_np[0].shape[0]
        if len(real_image_np) * batch_size >= fake_image_np.shape[0]:
            break
    real_image_np = np.concatenate(real_image_np, 0)[:fake_image_np.shape[0]]
    fid_score = calculate_fid(real_image_np, fake_image_np, batch_size=300)
    var_fid = fid_score[0][2]
    fid = round(fid_score[0][1], 3)
    print('------------------------fid_score--------------------------')
    print(fid_score)

    # Generate a batch of images
    sample_dir = os.path.join(args.path_helper['sample_path'], 'sample_dir')
    Path(sample_dir).mkdir(exist_ok=True)

    sample_imgs = gen_net(fixed_z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0)
    img_grid = make_grid(sample_imgs, nrow=5).to('cpu', torch.uint8).numpy()
    file_name = os.path.join(sample_dir, f'epoch_{epoch}_fid_{fid}.png')
    imsave(file_name, img_grid.swapaxes(0, 1).swapaxes(1, 2))

    writer_dict['valid_global_steps'] = global_steps + 1
    return inception_score, fid


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
