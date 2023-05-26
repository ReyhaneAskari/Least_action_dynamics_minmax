# CGD optim source code from:
# https://github.com/devzhk/Implicit-Competitive-Regularization
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from scipy.stats import gaussian_kde
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
import math
import os
import warnings
import time

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

iterations = 100
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


def conjugate_gradient(grad_x, grad_y,
                       x_params, y_params,
                       b, x=None, nsteps=10,
                       tol=1e-10, atol=1e-16,
                       lr_x=1.0, lr_y=1.0,
                       device=torch.device('cpu')):
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone().detach()
    else:
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=x, retain_graph=True).detach_().mul(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).detach_().mul(lr_x)
        Avx = x + h2
        r = b.clone().detach() - Avx

    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * rdotr

    for i in range(nsteps):
        # To compute Avp
        h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=p, retain_graph=True).detach_().mul(lr_y)
        h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).detach_().mul(lr_x)
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)

        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    if i > 99:
        warnings.warn('CG iter num: %d' % (i + 1))
    return x, i + 1


def Hvp_vec(grad_vec, params, vec, retain_graph=False):
    '''
    return Hessian vector product
    '''
    if torch.isnan(grad_vec).any():
        raise ValueError('Gradvec nan')
    if torch.isnan(vec).any():
        raise ValueError('vector nan')
        # zero padding for None
    grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec,
                              retain_graph=retain_graph,
                              allow_unused=True)
    grad_list = []
    for i, p in enumerate(params):
        if grad_grad[i] is None:
            grad_list.append(torch.zeros_like(p).view(-1))
        else:
            grad_list.append(grad_grad[i].contiguous().view(-1))
    hvp = torch.cat(grad_list)
    if torch.isnan(hvp).any():
        raise ValueError('hvp Nan')
    return hvp


def general_conjugate_gradient(grad_x, grad_y,
                               x_params, y_params, b,
                               lr_x, lr_y, x=None, nsteps=None,
                               tol=1e-12, atol=1e-20,
                               device=torch.device('cpu')):
    lr_x = lr_x.sqrt()
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone().detach()
    else:
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * x, retain_graph=True).mul_(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).mul_(lr_x)
        Avx = x + h2
        r = b.clone().detach() - Avx

    if nsteps is None:
        nsteps = b.shape[0]

    if grad_x.shape != b.shape:
        raise RuntimeError('CG: hessian vector product shape mismatch')
    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * rdotr
    for i in range(nsteps):
        # To compute Avp
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True)
        h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True).mul_(lr_y)
        # h_1.mul_(lr_y)
        # lr_y * D_yx * b
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=lr_y * h_1, retain_graph=True)
        h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).mul_(lr_x)
        # h_2.mul_(lr_x)
        # lr_x * D_xy * lr_y * D_yx * b
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    return x, i + 1


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()


class ACGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 eps=1e-5, beta=0.99,
                 tol=1e-12, atol=1e-20,
                 device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.state = {'lr_max': lr_max, 'lr_min': lr_min,
                      'eps': eps, 'solve_x': solve_x,
                      'tol': tol, 'atol': atol,
                      'beta': beta, 'step': 0,
                      'old_max': None, 'old_min': None,  # start point of CG
                      'sq_exp_avg_max': None, 'sq_exp_avg_min': None}  # save last update
        self.info = {'grad_x': None, 'grad_y': None,
                     'hvp_x': None, 'hvp_y': None,
                     'cg_x': None, 'cg_y': None,
                     'time': 0, 'iter_num': 0}
        self.device = device
        self.collect_info = collect_info

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def get_info(self):
        if self.info['grad_x'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
        print('Load state: {}'.format(state_dict))

    def set_lr(self, lr_max, lr_min):
        self.state.update({'lr_max': lr_max, 'lr_min': lr_min})
        print('Maximizing side learning rate: {:.4f}\n '
              'Minimizing side learning rate: {:.4f}'.format(lr_max, lr_min))

    def step(self, loss):
        lr_max = self.state['lr_max']
        lr_min = self.state['lr_min']
        beta = self.state['beta']
        eps = self.state['eps']
        tol = self.state['tol']
        atol = self.state['atol']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step

        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
        grad_x_vec_d = grad_x_vec.clone().detach()
        grad_y_vec_d = grad_y_vec.clone().detach()

        sq_avg_x = self.state['sq_exp_avg_max']
        sq_avg_y = self.state['sq_exp_avg_min']
        sq_avg_x = torch.zeros_like(grad_x_vec_d, requires_grad=False) if sq_avg_x is None else sq_avg_x
        sq_avg_y = torch.zeros_like(grad_y_vec_d, requires_grad=False) if sq_avg_y is None else sq_avg_y
        sq_avg_x.mul_(beta).addcmul_(1 - beta, grad_x_vec_d, grad_x_vec_d)
        sq_avg_y.mul_(beta).addcmul_(1 - beta, grad_y_vec_d, grad_y_vec_d)

        bias_correction = 1 - beta ** time_step
        lr_max = math.sqrt(bias_correction) * lr_max / sq_avg_x.sqrt().add(eps)
        lr_min = math.sqrt(bias_correction) * lr_min / sq_avg_y.sqrt().add(eps)

        scaled_grad_x = torch.mul(lr_max, grad_x_vec_d)
        scaled_grad_y = torch.mul(lr_min, grad_y_vec_d)
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, scaled_grad_y,
                            retain_graph=True)  # h_xy * d_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, scaled_grad_x,
                            retain_graph=True)  # h_yx * d_x
        p_x = torch.add(grad_x_vec_d, - hvp_x_vec)
        p_y = torch.add(grad_y_vec_d, hvp_y_vec)
        if self.collect_info:
            norm_px = torch.norm(hvp_x_vec, p=2).item()
            norm_py = torch.norm(hvp_y_vec, p=2).item()
            timer = time.time()

        if self.state['solve_x']:
            p_y.mul_(lr_min.sqrt())
            cg_y, iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                        x_params=self.min_params, y_params=self.max_params,
                                                        b=p_y, x=self.state['old_min'],
                                                        tol=tol, atol=atol,
                                                        lr_x=lr_min, lr_y=lr_max, device=self.device)
            old_min = cg_y.detach_()
            min_update = cg_y.mul(- lr_min.sqrt())
            hcg = Hvp_vec(grad_y_vec, self.max_params, min_update).detach_()
            hcg.add_(grad_x_vec_d)
            max_update = hcg.mul(lr_max)
            old_max = hcg.mul(lr_max.sqrt())
        else:
            p_x.mul_(lr_max.sqrt())
            cg_x, iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                        x_params=self.max_params, y_params=self.min_params,
                                                        b=p_x, x=self.state['old_max'],
                                                        tol=tol, atol=atol,
                                                        lr_x=lr_max, lr_y=lr_min, device=self.device)
            old_max = cg_x.detach_()
            max_update = cg_x.mul(lr_max.sqrt())
            hcg = Hvp_vec(grad_x_vec, self.min_params, max_update).detach_()
            hcg.add_(grad_y_vec_d)
            min_update = hcg.mul(- lr_min)
            old_min = hcg.mul(lr_min.sqrt())
        self.state.update({'old_max': old_max, 'old_min': old_min,
                           'sq_exp_avg_max': sq_avg_x, 'sq_exp_avg_min': sq_avg_y})

        if self.collect_info:
            timer = time.time() - timer
            self.info.update({'time': timer, 'iter_num': iter_num,
                              'hvp_x': norm_px, 'hvp_y': norm_py})

        index = 0
        for p in self.max_params:
            p.data.add_(max_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == max_update.numel(), 'Maximizer CG size mismatch'

        index = 0
        for p in self.min_params:
            p.data.add_(min_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == min_update.numel(), 'Minimizer CG size mismatch'

        if self.collect_info:
            norm_gx = torch.norm(grad_x_vec, p=2).item()
            norm_gy = torch.norm(grad_y_vec, p=2).item()
            norm_cgx = torch.norm(max_update, p=2).item()
            norm_cgy = torch.norm(min_update, p=2).item()
            self.info.update({'grad_x': norm_gx, 'grad_y': norm_gy,
                              'cg_x': norm_cgx, 'cg_y': norm_cgy})
        self.state['solve_x'] = False if self.state['solve_x'] else True


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

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

optimizer = ACGD(max_params=gen.parameters(), min_params=dis.parameters(),
                 device=device, solve_x=False, collect_info=False)

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

    gradient_penalty = 0

    loss_d = loss_real + loss_fake + gradient_penalty

    grad_d = torch.autograd.grad(
        loss_d, inputs=(dis.parameters()), create_graph=True)
    for p, g in zip(dis.parameters(), grad_d):
        p.grad = g
    optimizer.step(loss_d)

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

    optimizer.step(loss_g)

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    if iteration > 3:
        elapsed_time_list.append(elapsed_time_ms)
    print(elapsed_time_ms)

    print("iteration: " + str(iteration))

avg_time = np.mean(elapsed_time_list)

print('avg_time: ' + str(avg_time))
