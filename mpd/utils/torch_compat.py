import torch


def torch_load_compat(*args, **kwargs):
    if "weights_only" in kwargs:
        return torch.load(*args, **kwargs)

    try:
        return torch.load(*args, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(*args, **kwargs)
