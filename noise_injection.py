import torch


# this defines a noise injection that is an affine transformation around some centroid.
# if c = centroid, mu = mean, sigma = covariance matrix (diagonal), then the affine transformation is
# x = (sigma * c + mu) - c ; this effectively defines x as the distance between the centroid and the new point
# when c = latents this corresponds to a translation in the latent space
def affine_transform(sample_fn, fitness_fn, latent_shape, device, center = None, mean_scale=1):
    b, c, h, w = latent_shape
    # this is equal to the mean and covariance diagonal length
    # note we apply the transform on the channel dimension, not spatially. 
    solution_length = c * 2 
    centeroid = center or torch.zeros((b, c, h, w)).to(dtype=torch.float32, device=device)
    def _fitness(x):
        x = x.reshape(-1, c * 2)
        mean, cov_diag = x.chunk(2, dim=-1)
        mean =  mean.to(device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        cov_diag = cov_diag.to(device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        x = (cov_diag * centeroid + mean_scale * mean) - centeroid
        samples = sample_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)
    return _fitness, centeroid, solution_length