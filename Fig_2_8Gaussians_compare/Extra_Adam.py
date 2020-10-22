# ExtraAdam optim from https://github.com/GauthierGidel/Variational-Inequality-GAN
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
import torch.autograd as autograd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector
import math
import os

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

update_rule = 'extra_adam'
dis_iter = 1
_batch_size = 256
dim = 2000
use_cuda = True
z_dim = 64

iterations = 1000
lr = 3e-4
beta = 0.55
alpha = 0.6


class ExtraAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        self.params_copy = []

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(ExtraAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ExtraAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def extrapolation(self):
        # Check if a copy of the parameters was already made.
        is_empty = len(self.params_copy) == 0
        for group in self.param_groups:
            for p in group['params']:
                u = self.update(p, group)
                if is_empty:
                    # Save the current parameters for the update step. Several extrapolation step can be made before each update but only the parameters before the first extrapolation step are saved.
                    self.params_copy.append(p.data.clone())
                if u is None:
                    continue
                # Update the current parameters
                p.data.add_(u)

    def step(self, closure=None):
        if len(self.params_copy) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

        loss = None
        if closure is not None:
            loss = closure()

        i = -1
        for group in self.param_groups:
            for p in group['params']:
                i += 1
                u = self.update(p, group)
                if u is None:
                    continue
                # Update the parameters saved during the extrapolation step
                p.data = self.params_copy[i].add_(u)

        # Free the old parameters
        self.params_copy = []
        return loss

    def update(self, p, group):
        if p.grad is None:
            return None
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = group['amsgrad']

        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad = grad.add(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        return -step_size * exp_avg / denom


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
    # plt.show()
    if not os.path.exists('8_G_res/_' + name):
        os.makedirs('8_G_res/_' + name)
    plt.savefig('8_G_res/_' + name + '/' + str(epoch) + '.png')
    plt.close()

dis = Dis()
gen = Gen()

dis.apply(weights_init)
gen.apply(weights_init)

if use_cuda:
    dis = dis.cuda()
    gen = gen.cuda()

if update_rule == 'adam':
    dis_optimizer = Adam(dis.parameters(),
                         lr=lr,
                         betas=(beta, 0.9))
    gen_optimizer = Adam(gen.parameters(),
                         lr=lr,
                         betas=(0.5, 0.9))
elif update_rule == 'sgd':
    dis_optimizer = SGD(dis.parameters(), lr=0.01)
    gen_optimizer = SGD(gen.parameters(), lr=0.01)

elif update_rule == 'extra_adam':
    dis_optimizer = ExtraAdam(dis.parameters(), lr=lr, betas=(0.5, 0.9))
    gen_optimizer = ExtraAdam(gen.parameters(), lr=lr, betas=(0.5, 0.9))

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

    # extrapolation or the half step
    noise = torch.randn(_batch_size, z_dim)
    if use_cuda:
        noise = noise.cuda()

    noise = autograd.Variable(noise)
    real = dataset.__next__()
    loss_real = criterion(dis(real), ones)
    fake = gen(noise)
    loss_fake = criterion(dis(fake), zeros)

    gradient_penalty = 0

    loss_d = loss_real + loss_fake + gradient_penalty

    grad_d = torch.autograd.grad(
        loss_d, inputs=(dis.parameters()), create_graph=True)
    for p, g in zip(dis.parameters(), grad_d):
        p.grad = g
    dis_optimizer.extrapolation()

    noise = torch.randn(_batch_size, z_dim)
    ones = Variable(torch.ones(_batch_size))
    zeros = Variable(torch.zeros(_batch_size))
    if use_cuda:
        noise = noise.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    noise = autograd.Variable(noise)
    fake = gen(noise)
    loss_g = criterion(dis(fake), ones)
    grad_g = torch.autograd.grad(
        loss_g, inputs=(gen.parameters()), create_graph=True)
    for p, g in zip(gen.parameters(), grad_g):
        p.grad = g
    gen_optimizer.extrapolation()

    # full step
    noise = torch.randn(_batch_size, z_dim)
    if use_cuda:
        noise = noise.cuda()

    noise = autograd.Variable(noise)
    real = dataset.__next__()
    loss_real = criterion(dis(real), ones)
    fake = gen(noise)
    loss_fake = criterion(dis(fake), zeros)

    gradient_penalty = 0

    loss_d = loss_real + loss_fake + gradient_penalty

    grad_d = torch.autograd.grad(
        loss_d, inputs=(dis.parameters()), create_graph=True)
    for p, g in zip(dis.parameters(), grad_d):
        p.grad = g
    dis_optimizer.step()

    noise = torch.randn(_batch_size, z_dim)
    ones = Variable(torch.ones(_batch_size))
    zeros = Variable(torch.zeros(_batch_size))
    if use_cuda:
        noise = noise.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    noise = autograd.Variable(noise)
    fake = gen(noise)
    loss_g = criterion(dis(fake), ones)
    grad_g = torch.autograd.grad(
        loss_g, inputs=(gen.parameters()), create_graph=True)
    for p, g in zip(gen.parameters(), grad_g):
        p.grad = g
    gen_optimizer.step()

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    if iteration > 3:
        elapsed_time_list.append(elapsed_time_ms)
    print(elapsed_time_ms)

    print("iteration: " + str(iteration))

avg_time = np.mean(elapsed_time_list)

print('avg_time: ' + str(avg_time))
