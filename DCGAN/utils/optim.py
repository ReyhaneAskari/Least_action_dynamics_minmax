import torch
from torch import optim
import math


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


class VJP_Adam(optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, alpha_vjp=0.0, alpha_grad=1.0):
            if not 0.0 <= lr:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if not 0.0 <= eps:
                raise ValueError("Invalid epsilon value: {}".format(eps))
            # if not 0.0 <= betas[0] < 1.0:
            #     raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
            # if not 0.0 <= betas[1] < 1.0:
            #     raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            defaults = dict(lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay, amsgrad=amsgrad,
                            alpha_vjp=alpha_vjp, alpha_grad=alpha_grad)
            super(VJP_Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VJP_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, vjps, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for index, p in enumerate(group['params']):
                vjp = vjps[index]
                if p.grad is None:
                    continue
                # vjp_new = p.grad.data.norm(2) * vjp / (vjp.norm(2) + 1e-5)
                grad = group['alpha_grad'] * p.grad.data - vjp * group['alpha_vjp']
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

                # if GLOB:
                #     import ipdb; ipdb.set_trace()

                p.data.addcdiv_(-step_size, exp_avg, denom)
                # p.data = p.data + group['alpha'] * vjp

        return loss
