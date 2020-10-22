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
import os

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

update_rule = 'adam'
dis_iter = 1
_batch_size = 256
dim = 2000
use_cuda = True
z_dim = 64

iterations = 1000
lr = 3e-4
beta = 0.55
alpha = 0.6
LAMBDA = 0.5


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


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(_batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(
            disc_interpolates.size()).cuda() if use_cuda else torch.ones(
            disc_interpolates.size()),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


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

    gradient_penalty = calc_gradient_penalty(dis, real.data, fake.data)

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
