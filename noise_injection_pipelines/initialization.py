import torch

def rotation_initalization(values, solution_length, latents_shape, stdev=0.1):
    _, c, h, w = latents_shape
    identity = torch.eye(c, device=values.device, dtype=values.dtype).reshape(1, -1)
    values[:, :solution_length - c] = identity
    values +=  torch.randn(values.shape, device=values.device, dtype=values.dtype) * stdev
    return values

def randn_intialization(values, mean=0, stdev=1):
    values.normal_(mean, stdev)
