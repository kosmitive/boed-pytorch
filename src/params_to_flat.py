import torch
import numpy as np


def params_to_flat(p, params):

    # arange in vector
    cnt = 0
    for g, pa in zip(p, params):
        v =  g.contiguous().view(-1) if g is not None else torch.zeros(np.prod(pa.shape))
        h_vector = v if cnt == 0 else torch.cat(
            [h_vector, v])
        cnt = 1

    return h_vector
