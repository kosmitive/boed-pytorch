"""Flatten a parameter-aligned gradient tuple into one vector."""

from collections.abc import Iterable, Sequence

import numpy as np
import torch


def params_to_flat(
    grads: Sequence[torch.Tensor | None],
    params: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Concatenate gradients in parameter order.

    Args:
        grads: One gradient per parameter. ``None`` is treated as a zero
            gradient of the matching shape.
        params: Parameters aligned with ``grads``.

    Returns:
        A one-dimensional vector of every gradient entry.
    """
    flat: torch.Tensor | None = None
    for grad, param in zip(grads, params):
        if grad is None:
            piece = torch.zeros(
                int(np.prod(tuple(param.shape))),
                device=param.device,
                dtype=param.dtype,
            )
        else:
            piece = grad.contiguous().view(-1)
        flat = piece if flat is None else torch.cat([flat, piece])
    if flat is None:
        return torch.zeros(0)
    return flat
