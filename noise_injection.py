import torch


def rotational_transform(sample_fn, fitness_fn, latent_shape, device, center = None, mean_scale=1):
    b, c, h, w = latent_shape
    # this is equal to the mean and covariance diagonal length
    # note we apply the transform on the channel dimension, not spatially. 
    solution_length = c + c ** 2
    centroid = center if center is not None else torch.zeros((b, c, h, w)).to(dtype=torch.float32, device=device)
    def _fitness(x):
        x = x.reshape(-1, c + c ** 2)
        mean = x[:, :c]
        cov = x[:, c:].reshape(-1, c, c)
        rot, scaling = torch.linalg.qr(cov)
        mean =  mean.to(device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=torch.float32).unsqueeze(-1)
        rot = rot.permute(0, 3, 1, 2)
        x = centroid.permute(0, 2, 3, 1) @ rot 
        x = x.permute(0, 3, 1, 2) + mean_scale * mean - centroid
        samples = sample_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)
    return _fitness, centroid, solution_length
