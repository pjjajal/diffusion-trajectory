import torch
from einops import einsum

# this function parameterizes the noise as the difference between the orginal latent and a rotated + translated version of it.
def rotational_transform(sample_fn, fitness_fn, latent_shape, device, center = None, mean_scale=1):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = c + c ** 2
    centroid = center if center is not None else torch.zeros((b, c, h, w)).to(dtype=torch.float32, device=device)
    def _fitness(x):
        x = x.reshape(-1, c + c ** 2)
        # slice out the mean and covariance matrix
        mean = x[:, :c]
        cov = x[:, c:].reshape(-1, c, c)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean =  mean.to(device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=torch.float32).unsqueeze(-1)
        rot = rot.permute(0, 3, 1, 2)
        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = centroid.permute(0, 2, 3, 1) @ rot 
        x = x.permute(0, 3, 1, 2) + mean_scale * mean - centroid
        samples = sample_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)
    return _fitness, centroid, solution_length


def rotational_transform_all(sample_fn, fitness_fn, latent_shape, device, center = None, mean_scale=1):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = (c + c ** 2) * h * w
    centroid = center if center is not None else torch.zeros((b, c, h, w)).to(dtype=torch.float32, device=device)
    def _fitness(x):
        x = x.reshape(-1, c + c ** 2, h, w)
        # slice out the mean and covariance matrix
        mean = x[:, :c, :, :]
        cov = x[:, c:, :, :].reshape(-1, c, c, h, w)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean =  mean.to(device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=torch.float32).unsqueeze(-1)
        rot = rot.permute(0, 3, 1, 2)
        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(rot, centroid, 'p ... c1 c, b c h w  -> p h w c1')
        x = x.permute(0, 3, 1, 2) + mean_scale * mean - centroid
        samples = sample_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)
    return _fitness, centroid, solution_length