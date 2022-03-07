from typing import Callable, TypeVar

import torch
import torch.nn.functional as F


Criterion = TypeVar(
    'Criterion',
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
)


l1_criterion = lambda r, gt: F.l1_loss(r, 1 - gt)
l2_criterion = lambda r, gt: F.mse_loss(r, 1 - gt)
bce_criterion = lambda r, gt: F.binary_cross_entropy(r, 1 - gt)
fcon_criterion = lambda r: sum(f.transpose(0, 1).flatten(1).std(-1).mean() for f in r).div_(len(r))
