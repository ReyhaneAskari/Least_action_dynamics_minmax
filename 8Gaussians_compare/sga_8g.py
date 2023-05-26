# SGA PyTorch implemented based on the official
# DeepMind implementation:
# https://github.com/deepmind/symplectic-gradient-adjustment/blob/master/Symplectic_Gradient_Adjustment.ipynb

import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from scipy.stats import gaussian_kde
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
import math
from itertools import chain
import os

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

update_rule = 'adam_sga'
dis_iter = 1
_batch_size = 256
dim = 2000
use_cuda = True
z_dim = 64

iterations = 1000
lr = 3e-4
beta = 0.55
alpha = 0.6


def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


def parameters_grad_to_vector(parameters):
    param_device = None

    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        vec.append(param.grad.view(-1))
    return torch.cat(vec)


class RMSprop(optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0, centered=False):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    torch.no_grad()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = square_avg.sqrt().add_(group['eps'])
                p.addcdiv_(grad, avg, value=-group['lr'])

        return loss


class Gen(nn.Module):

    def __init__(self):
        super(Gen, self).__init__()

        main = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 2),
        )
        self.main = main

    def forward(self, noise):
            output = self.main(noise)
            return output


class Dis(nn.Module):

    def __init__(self):
        super(Dis, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_8gaussians(batch_size):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * .05
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414
        out = Variable(torch.Tensor(dataset))
        if use_cuda:
            out = out.cuda()
        yield out


def get_dens_real(batch_size):
    data = get_8gaussians(batch_size).__next__()
    real = np.array(data.data.cpu())
    kde_real = gaussian_kde(real.T, bw_method=0.22)
    x, y = np.mgrid[-2:2:(200 * 1j), -2:2:(200 * 1j)]
    z_real = kde_real((x.ravel(), y.ravel())).reshape(*x.shape)
    return z_real

z_real = get_dens_real(1000)


def plot(fake, epoch, name):
    plt.figure(figsize=(20, 9))
    fake = np.array(fake.data.cpu())
    kde_fake = gaussian_kde(fake.T, bw_method=0.22)

    x, y = np.mgrid[-2:2:(200 * 1j), -2:2:(200 * 1j)]
    z_fake = kde_fake((x.ravel(), y.ravel())).reshape(*x.shape)

    ax1 = plt.subplot(1, 2, 1)
    ax1.pcolor(x, y, z_real, cmap='GnBu')

    ax2 = plt.subplot(1, 2, 2)
    ax2.pcolor(x, y, z_fake, cmap='GnBu')
    ax1.scatter(real.data.cpu().numpy()[:, 0],
                real.data.cpu().numpy()[:, 1])
    ax2.scatter(fake[:, 0], fake[:, 1])
    if not os.path.exists('8_G_res/_' + name):
        os.makedirs('8_G_res/_' + name)
    plt.savefig('8_G_res/_' + name + '/' + str(epoch) + '.png')

dis = Dis()
gen = Gen()

dis.apply(weights_init)
gen.apply(weights_init)

if use_cuda:
    dis = dis.cuda()
    gen = gen.cuda()

optimizer = optim.RMSprop(
    list(dis.parameters()) + list(gen.parameters()), lr=lr, alpha=0.9)
one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

dataset = get_8gaussians(_batch_size)
criterion = nn.BCEWithLogitsLoss()

ones = Variable(torch.ones(_batch_size))
zeros = Variable(torch.zeros(_batch_size))
if use_cuda:
    criterion = criterion.cuda()
    ones = ones.cuda()
    zeros = zeros.cuda()

points = []
dis_params_flatten = parameters_to_vector(dis.parameters())
gen_params_flatten = parameters_to_vector(gen.parameters())

# just to fill the empty grad buffers
noise = torch.randn(_batch_size, z_dim)
if use_cuda:
    noise = noise.cuda()
noise = autograd.Variable(noise)
fake = gen(noise)
pred_fake = criterion(dis(fake), zeros).sum()
(0.0 * pred_fake).backward(create_graph=True)
gen_loss = 0
pred_tot = 0
elapsed_time_list = []

for iteration in range(iterations):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    noise = torch.randn(_batch_size, z_dim)
    if use_cuda:
        noise = noise.cuda()

    noise = autograd.Variable(noise)
    real = dataset.__next__()
    loss_real = criterion(dis(real), ones)
    fake = gen(noise)
    loss_fake = criterion(dis(fake), zeros)

    loss_d = loss_real + loss_fake
    grad_d = torch.autograd.grad(
        loss_d, inputs=(dis.parameters()), create_graph=True)

    loss_g = criterion(dis(fake), ones)
    grad_g = torch.autograd.grad(
        loss_g, inputs=(gen.parameters()), create_graph=True)

    for p, g in zip(dis.parameters(), grad_d):
        p.grad = g
    for p, g in zip(gen.parameters(), grad_g):
        p.grad = g

    grad_dis_params_flatten = parameters_grad_to_vector(dis.parameters())
    grad_gen_params_flatten = parameters_grad_to_vector(gen.parameters())

    grads = torch.cat([grad_dis_params_flatten, grad_gen_params_flatten])
    params = list(dis.parameters()) + list(gen.parameters())
    # H.T * grads  where   H = d grad / d params
    HTg = torch.autograd.grad(
        grads, params, grad_outputs=grads, create_graph=True)
    HTg_flatten = parameters_to_vector(HTg)
    # grads * H.T   equivalent to   H * grads
    Hg_flatten = autograd.grad(
        HTg_flatten, grads, grad_outputs=grads.detach(), create_graph=True)[0]

    ATg = (HTg_flatten - Hg_flatten) / 2.0

    grad_dot_h = (grads * HTg_flatten).sum()
    ATg_dot_h = (ATg * HTg_flatten).sum()
    mult = grad_dot_h * ATg_dot_h
    reg_params = 1.0
    lambda_ = torch.sign(mult / ATg.shape[0] + 0.1) * reg_params
    grads = grads + lambda_ * ATg

    place_holder = [p.data for p in params]
    torch.nn.utils.vector_to_parameters(grads, place_holder)
    for j, p in enumerate(params):
        p.grad.data = place_holder[j]

    optimizer.step()

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    if iteration > 3:
        elapsed_time_list.append(elapsed_time_ms)
    print(elapsed_time_ms)

    print("iteration: " + str(iteration))

avg_time = np.mean(elapsed_time_list)

print('avg_time: ' + str(avg_time))
