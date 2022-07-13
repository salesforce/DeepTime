from typing import Optional, Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor


def get_loss_fn(loss_name: str,
                delta: Optional[float] = 1.0,
                beta: Optional[float] = 1.0) -> Callable:
    return {'mse': F.mse_loss,
            'mae': F.l1_loss,
            'huber': partial(F.huber_loss, delta=delta),
            'smooth_l1': partial(F.smooth_l1_loss, beta=beta)}[loss_name]
