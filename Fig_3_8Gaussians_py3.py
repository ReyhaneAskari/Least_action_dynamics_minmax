import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
import torch.autograd as autograd
from scipy.stats import gaussian_kde
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
import math

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

update_rule = 'adam_lsd'
dis_iter = 1
_batch_size = 256
dim = 2000
use_cuda = True
z_dim = 64

loss_type = 'non_zero_sum'
iterations = 5001
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


class LSD_Adam(optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, alpha=0.0):
            if not 0.0 <= lr:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if not 0.0 <= eps:
                raise ValueError("Invalid epsilon value: {}".format(eps))
            defaults = dict(lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay, amsgrad=amsgrad, alpha=alpha)
            super(LSD_Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LSD_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, vjps, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for index, p in enumerate(group['params']):
                vjp = vjps[index]
                if p.grad is None:
                    continue
                grad = p.grad.data - group['alpha'] * vjp
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
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss


class MLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(MLinear, self).__init__(*args, **kwargs)
        self.m = torch.nn.Parameter(self.weight.norm(2, dim=1, keepdim=True))

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


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
    plt.show()

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

elif update_rule == 'adam_lsd':
    dis_optimizer = LSD_Adam(dis.parameters(), lr=lr, betas=(beta, 0.9),
                             alpha=alpha)
    gen_optimizer = LSD_Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9),
                             alpha=alpha)

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

for iteration in range(iterations):
    for iter_d in range(dis_iter):
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

        if update_rule == 'adam_lsd':
            gen_params_flatten_prev = gen_params_flatten + 0.0
            gen_params_flatten = parameters_to_vector(gen.parameters()) + 0.0
            grad_gen_params_flatten = parameters_grad_to_vector(gen.parameters())
            delta_gen_params_flatten = gen_params_flatten - gen_params_flatten_prev

            vjp_dis = torch.autograd.grad(
                grad_gen_params_flatten, dis.parameters(),
                grad_outputs=delta_gen_params_flatten)

            dis_optimizer.step(vjps=vjp_dis)
        else:
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
    if loss_type == 'zero_sum':
        loss_g = -criterion(dis(fake), zeros)
    else:
        loss_g = criterion(dis(fake), ones)
    grad_g = torch.autograd.grad(
        loss_g, inputs=(gen.parameters()), create_graph=True)
    for p, g in zip(gen.parameters(), grad_g):
        p.grad = g

    if update_rule == 'adam_lsd':
        dis_params_flatten_prev = dis_params_flatten + 0.0
        dis_params_flatten = parameters_to_vector(dis.parameters())
        grad_dis_params_flatten = parameters_grad_to_vector(dis.parameters())
        delta_dis_params_flatten = dis_params_flatten - dis_params_flatten_prev

        vjp_gen = torch.autograd.grad(
            grad_dis_params_flatten, gen.parameters(),
            grad_outputs=delta_dis_params_flatten)
        gen_optimizer.step(vjps=vjp_gen)
    else:
        gen_optimizer.step()

    print("iteration: " + str(iteration) +
          " gen_loss: " + str(float(loss_g)) +
          " dis_loss: " + str(float(loss_d)))
    if iteration % 100 == 99:
        noise = torch.randn(1000, z_dim)
        if use_cuda:
            noise = noise.cuda()
        noise = autograd.Variable(noise)
        fake_for_plot = gen(noise)
        plot(fake_for_plot, iteration,
             name=(str(update_rule) +
                   '_lr_' + str(lr) +
                   '_loss_type_' + str(loss_type) +
                   '_beta_' + str(beta) +
                   '_alpha_' + str(alpha)))
