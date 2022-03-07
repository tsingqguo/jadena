from math import inf

import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.stats as st

from .attack import Attack
from .criterions import Criterion

def _gaussian_kernel(kernel_size, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernel_size)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

class BiasFieldAttack(Attack):
    def __init__(
        self,
        model: nn.Module,
        criterion: Criterion,
        step: int,
        attack_mode: str = 'first',
        noise_mode: str = 'none',
        bias_mode: str = 'none',
        spatial_mode: str = 'none',
        noise_lr: float = 0.,
        bias_lr: float = 0.,
        spatial_lr: float = 0.,
        lambda_l: float = 1.,
        lambda_n: float = 0.,
        lambda_b: float = 0.,
        lambda_s: float = 0.,
        momentum_decay: float = 0.,
        epsilon: float = inf,
        epsilon_n: float = inf,
        noise_ti: bool = False,
        degree: int = 10,
        tune_scale: int = 8,
        ti_size: int = 21,
    ):
        super(BiasFieldAttack, self).__init__(model)
        assert step >= 0, f'step should be non-negative integer, got {step}'
        assert attack_mode in ('first', 'all')
        assert noise_mode in ('none', 'add')
        assert bias_mode in ('none', 'rgb', 'same')
        assert spatial_mode in ('none', 'spatial_weight', 'super_pixel', 'optical_flow')
        assert noise_lr >= 0, f'noise_lr should be non-negative floats, got {noise_lr}'
        assert bias_lr >= 0, f'bias_lr should be non-negative floats, got {bias_lr}'
        assert spatial_lr >= 0, f'spatial_lr should be non-negative floats, got {spatial_lr}'
        assert lambda_l >= 0, f'lambda_l should be non-negative float, got {lambda_l}'
        assert lambda_n >= 0, f'lambda_b should be non-negative float, got {lambda_n}'
        assert lambda_b >= 0, f'lambda_b should be non-negative float, got {lambda_b}'
        assert lambda_s >= 0, f'lambda_s should be non-negative float, got {lambda_s}'
        assert momentum_decay >= 0, f'momentum_decay should be non-negative float, got {momentum_decay}'
        assert epsilon >= 0, f'epsilon should be non-negative float, got {epsilon}'
        assert epsilon_n >= 0, f'epsilon_n should be non-negative float, got {epsilon_n}'
        assert degree > 0, f'degree should be positive integer, got {degree}'
        assert tune_scale > 0, f'tune_scale should be positive integer, got {tune_scale}'
        assert ti_size > 0, f'ti_size should be positive integer, got {ti_size}'
        if noise_ti:
            assert noise_mode == 'add'
        self.criterion = criterion
        self.step = step
        self.attack_mode = attack_mode
        self.noise_mode = noise_mode
        self.bias_mode = bias_mode
        self.spatial_mode = spatial_mode
        self.noise_lr = noise_lr
        self.bias_lr = bias_lr
        self.spatial_lr = spatial_lr
        self.lambda_l = lambda_l
        self.lambda_n = lambda_n
        self.lambda_b = lambda_b
        self.lambda_s = lambda_s
        self.momentum_decay = momentum_decay
        self.epsilon = epsilon
        self.epsilon_n = epsilon_n
        self.noise_ti = noise_ti
        self.degree = degree
        self.tune_scale = tune_scale
        self.ti_size = ti_size

    def __call__(self, tensor: torch.Tensor, target: torch.Tensor = None, vis=False):
        # identity bias field
        n, c, h, w = tensor.size()
        assert c == 3, f'tensor should be batched RGB images, got {c} channels'

        params = []
        lrs = []

        if self.noise_mode == 'add':
            noise = torch.zeros_like(tensor).requires_grad_()
            params.append(noise)
            lrs.append(self.noise_lr)
        elif self.noise_mode == 'none':
            noise = None

        num_coef = (self.degree + 1) * (self.degree + 2) // 2
        if self.bias_mode in ('rgb', 'same'):
            coef = torch.zeros(n, 1 if self.bias_mode == 'same' else 3, num_coef).to(tensor).requires_grad_()
            params.append(coef)
            lrs.append(self.bias_lr)
        elif self.bias_mode == 'none':
            coef = None

        if self.spatial_mode == 'spatial_weight':
            coef_spatial = torch.zeros(n, num_coef).to(tensor).requires_grad_()
            params.append(coef_spatial)
            lrs.append(self.spatial_lr)
        elif self.spatial_mode == 'optical_flow':
            optical_flow = torch.zeros(n, h // self.tune_scale, w // self.tune_scale, 2).to(tensor).requires_grad_()
            params.append(optical_flow)
            lrs.append(self.spatial_lr)

        if self.noise_ti:
            ti_kernel = _gaussian_kernel(self.ti_size, 3)
            ti_kernel = torch.as_tensor(ti_kernel).to(tensor).expand(c, 1, -1, -1)
            ti_padding = (self.ti_size - 1) // 2

        # create coord base
        coord_x = torch.linspace(-1, 1, w).to(tensor)[None, :]
        coord_y = torch.linspace(-1, 1, h).to(tensor)[:, None]
        coord = torch.stack((coord_x.expand(h, -1), coord_y.expand(-1, w)), dim=-1)
        base = torch.zeros(num_coef, h, w).to(tensor)
        i = 0
        for t in range(self.degree + 1):
            for l in range(self.degree - t + 1):
                base[i, :, :].add_(coord_x ** t).mul_(coord_y ** l)
                if vis and (t <= 1 and l <= 1):
                    base[i, :, :] = 0
                i += 1
        del i

        if vis:
            step_perts_wo_noise = []
            step_perts = []
            step_exposures_wo_tuning = []
            step_exposures = []
            step_noises = []

        for n_iter in range(self.step + 1):
            pert = tensor.clone()
            # apply bias field
            if self.bias_mode in ('rgb', 'same'):
                bias_field = (base[None, None, :, :, :] * coef[:, :, :, None, None]).sum(dim=2)

                if vis and self.bias_mode != 'none':
                    step_exposures_wo_tuning.append(bias_field.detach().clone())

                if self.spatial_mode == 'spatial_weight':
                    spatial_tuning = (base[None, None, :, :, :] * coef_spatial[:, None, :, None, None]).sum(dim=2).sigmoid_()
                    bias_field = bias_field.mul_(spatial_tuning)
                elif self.spatial_mode == 'optical_flow':
                    upsampled_optical_flow = F.interpolate(
                        optical_flow.permute(0, 3, 1, 2),
                        align_corners=False,
                        mode='bilinear',
                        size=(h, w),
                    ).permute(0, 2, 3, 1)
                    spatial_tuning = (coord + upsampled_optical_flow).clamp_(-1, 1)
                    bias_field = F.grid_sample(bias_field, spatial_tuning, align_corners=True)
                elif self.spatial_mode == 'none':
                    spatial_tuning = None

                pert = pert.log_().add_(bias_field).exp_()

            if vis:
                if self.bias_mode != 'none':
                    step_exposures.append(bias_field.detach().clone())
                if self.noise_mode != 'none':
                    step_noises.append(noise.detach().clone())
                step_perts_wo_noise.append(pert.detach().clone())

            if self.noise_mode == 'add':
                pert = pert + noise

            # optimized clamp
            if self.epsilon != inf:
                pert = torch.min(pert, tensor + self.epsilon)
                pert = torch.max(pert, tensor - self.epsilon)
            pert = pert.clamp(0, 1)

            if vis:
                step_perts.append(pert.detach().clone())

            pred = self.model(pert)

            # calculate loss and sparsity constraint terms
            loss = torch.zeros(1).to(tensor)

            if self.lambda_l > 0:
                crit = self.criterion(pred) if target is None else self.criterion(pred, target)
                loss.add_(crit, alpha=self.lambda_l)

            if self.lambda_n > 0 and self.noise_mode != 'none':
                sparsity_n = noise.pow(2).sum()
                loss.add_(sparsity_n, alpha=self.lambda_n)

            if self.lambda_b > 0 and self.bias_mode != 'none':
                sparsity_b = bias_field.pow(2).sum().div_(h * w)
                loss.add_(sparsity_b, alpha=self.lambda_b)

            if self.lambda_s > 0 and self.spatial_mode == 'optical_flow':
                diff_s_h = (optical_flow[:, :, 1:, :] - optical_flow[:, :, :-1, :])
                diff_s_v = (optical_flow[:, 1:, :, :] - optical_flow[:, :-1, :, :])
                sparsity_s = diff_s_h.pow(2).sum() + diff_s_v.pow(2).sum()
                loss.add_(sparsity_s, alpha=self.lambda_s / (self.tune_scale * self.tune_scale))

            # grad and update
            if n_iter < self.step:
                with torch.no_grad():
                    grads = ag.grad(loss, params)

                    # apply ti on noise grad
                    if self.noise_ti:
                        grads = list(grads)
                        grads[0] = F.conv2d(grads[0], ti_kernel, padding=ti_padding, groups=c)

                    if self.momentum_decay > 0:
                        grad_norms = [grad.flatten(1).norm(p=1, dim=1) for grad in grads]
                        grad_norms = [norm.where(norm > 0, torch.ones(1).to(norm)) for norm in grad_norms]
                        grads = [
                            grad.div_(norm.view((-1, *((1,) * (grad.dim() - 1)))))
                            for grad, norm in zip(grads, grad_norms)
                        ]
                        if n_iter == 0:
                            cum_grads = grads
                        else:
                            cum_grads = [
                                cum_grad.mul_(self.momentum_decay).add_(grad)
                                for cum_grad, grad in zip(cum_grads, grads)
                            ]
                    else:
                        cum_grads = grads

                    for param, cum_grad, lr in zip(params, cum_grads, lrs):
                        if self.attack_mode == 'first':
                            param[0].sub_(cum_grad[0].sign(), alpha=lr)
                        elif self.attack_mode == 'all':
                            param.sub_(cum_grad.sign(), alpha=lr)
                    if self.noise_mode != 'none' and self.epsilon_n != inf:
                        noise = noise.clamp_(-self.epsilon_n, self.epsilon_n)

        extra = {'pred': pred}
        if self.noise_mode != 'none':
            extra['noise'] = noise
        if self.bias_mode != 'none':
            extra['bias_field'] = bias_field
        if self.spatial_mode != 'none':
            extra['spatial_tuning'] = spatial_tuning
        if vis:
            extra['step_perts_wo_noise'] = step_perts_wo_noise
            extra['step_perts'] = step_perts
            extra['step_exposures_wo_tuning'] = step_exposures_wo_tuning
            extra['step_exposures'] = step_exposures
            extra['step_noises'] = step_noises
        return pert, extra
