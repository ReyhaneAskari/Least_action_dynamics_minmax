import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.optimizer import required
np.random.seed(11)


iter_num = 100
x_init = np.sqrt(2) / 2.0
y_init = np.sqrt(2) / 2.0
beta_lsd = 0.5
lr_lsd = 0.1
alpha = 0.51
lyapunov_coef = 7.7


def lyapunov(x):
    return lyapunov_coef * (x[0] ** 2 + x[1] ** 2)


def f(x):
    return x[0] * x[1]


class LSDOpt(optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, alpha=0.1):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha)
        super(LSDOpt, self).__init__(params, defaults)

    def step(self, vjps, effect=[1, 1], closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']

            for index, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                vjp = vjps[index]
                grad = p.grad.data

                param_state = self.state[p]
                if 'previous_iterate' not in param_state:
                    prev_p = param_state['previous_iterate'] = p.data.detach()
                else:
                    prev_p = param_state['previous_iterate']

                param_state['previous_iterate'] = p.data

                p.data = p.data + effect[index] * momentum * (
                    p.data - prev_p.data) + group['alpha'] * vjp - group['lr'] * grad

        return loss


def LSD(net):
    net.x.data = torch.FloatTensor([x_init])
    net.y.data = torch.FloatTensor([y_init])
    xys = [[net.x.data[0] + 0, net.y.data[0] + 0]]
    vs = [[0.0, 0.0]]
    lyapunov_values = []

    opt = LSDOpt(net.parameters(), lr=lr_lsd, momentum=beta_lsd, alpha=alpha)
    for i in range(iter_num):
        xys += [[net.x.data[0], net.y.data[0]]]
        # update x
        loss = net()
        opt.zero_grad()
        loss.backward(create_graph=True)
        _, y_t = xys[-1]
        _, y_tm1 = xys[-2]
        v_y = (y_t - y_tm1)
        vjp_x = torch.autograd.grad(
            -net.y.grad, net.x,
            grad_outputs=torch.FloatTensor([v_y]),
            create_graph=True)[0]
        vjps = [vjp_x.data, 0]
        net.y.grad.data *= 0
        opt.step(vjps=vjps, effect=[1, 0])

        # update y
        loss = -net()
        opt.zero_grad()
        loss.backward(create_graph=True)
        x_t, _ = xys[-1]
        x_tm1, _ = xys[-2]
        v_x = x_t - x_tm1
        vjp_y = torch.autograd.grad(
            -net.x.grad, net.y,
            grad_outputs=torch.FloatTensor([v_x]),
            create_graph=True)[0]
        vjps = [0, vjp_y.data]
        net.x.grad.data *= 0
        opt.step(vjps=vjps, effect=[0, 1])
        vs += [[v_x, v_y]]
        lyapunov_values += [2 * (v_x**2 + v_y**2) + lyapunov_coef *
                            (net.x.data[0]**2 + net.y.data[0]**2)]
    xys = np.array(xys)
    vs = np.array(vs)
    lyapunov_values = np.array(lyapunov_values)
    return (xys, vs, lyapunov_values)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.x = torch.nn.Parameter(torch.FloatTensor([0]))
        self.y = torch.nn.Parameter(torch.FloatTensor([0]))

    def forward(self, xy=None):
        if xy is not None:
            x = xy[:, 0]
            y = xy[:, 1]
            return (x * y)
        return (self.x * self.y)

net = Net()
xys_vjp, vs_vjp, lyapunov_values = LSD(net)


def plot_lyapunov_descent(lyapunov_values):
    plt.figure(figsize=(5, 5))
    plt.plot(lyapunov_values, color='#0A4D8C')
    plt.show()
    plt.close()

plot_lyapunov_descent(lyapunov_values)


def plot_2_D_lyapunov(xys_vjp):
    x_lim = 1.0
    y_lim = 1.0
    x_0 = np.linspace(-x_lim, x_lim, 100)
    x_1 = np.linspace(-y_lim, y_lim, 100)
    zl = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            zl[j, i] = lyapunov(np.array(
                [x_0[i], x_1[j]])) + vs_vjp[i][0]**2 + vs_vjp[i][1]**2
    X_0, X_1 = np.meshgrid(x_0, x_1)
    cs = plt.contourf(X_0, X_1, zl, 15, cmap='Blues')
    plt.colorbar()
    plt.contour(cs, color='k', linewidths=0.3, ls='-', alpha=0.7)
    zs = []
    for i in range(xys_vjp.shape[0]):
        zs += [f(xys_vjp[i]) + np.linalg.norm(vs_vjp[i])]
    zs = np.array(zs)
    plt.plot(xys_vjp[:, 0], xys_vjp[:, 1], c='#DD1321', zorder=5)
    plt.show()

plot_2_D_lyapunov(xys_vjp)
