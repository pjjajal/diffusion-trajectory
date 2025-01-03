import torch
from einops import einsum

# this function parameterizes the noise as the difference between the orginal latent and a rotated + translated version of it.
def rotational_transform(sample_fn, fitness_fn, latent_shape, device, center = None, mean_scale=1, dtype=torch.float32):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = c + c ** 2
    centroid = center if center is not None else torch.zeros((b, c, h, w)).to(dtype=dtype, device=device)
    def _fitness(x):
        x = x.reshape(-1, c + c ** 2)
        # slice out the mean and covariance matrix
        mean = x[:, :c]
        cov = x[:, c:].reshape(-1, c, c)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean =  mean.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=dtype)
        # rot = rot.permute(0, 3, 1, 2)
        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(centroid, rot, 'b c h w, p c1 c -> p c1 h w')
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)
    return _fitness, centroid, solution_length


# this is another version of the rotational transform, where we have two means and a covariance matrix.
# the centroid is shifted (mean_2), rotated, and shifted (mean) again.
# this seems to work okay, results are not as good as rotation + translation.
def rotational_transform_double_shift(sample_fn, fitness_fn, latent_shape, device, center = None, mean_scale=1, mean_scale_2=1, dtype=torch.float32):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = 2 * c + c ** 2
    centroid = center if center is not None else torch.zeros((b, c, h, w)).to(dtype=dtype, device=device)
    def _fitness(x):
        x = x.reshape(-1, 2 * c + c ** 2)
        # slice out the mean and covariance matrix
        mean = x[:, :c]
        mean_2 = x[:, c:2*c]
        cov = x[:, 2*c:].reshape(-1, c, c)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean =  mean.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        mean_2 =  mean_2.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=dtype)
        # rot = rot.permute(0, 3, 1, 2)
        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(centroid + mean_scale_2 * mean_2, rot, 'b c h w, p c1 c -> p c1 h w')
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)
    return _fitness, centroid, solution_length



# this one does not work at all, do not try to use it.
def rotational_transform_all(sample_fn, fitness_fn, latent_shape, device, center = None, mean_scale=1, dtype=torch.float32):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = (c + c ** 2) * h * w
    centroid = center if center is not None else torch.zeros((b, c, h, w)).to(dtype=dtype, device=device)
    def _fitness(x):
        x = x.reshape(-1, c + c ** 2, h, w)
        # slice out the mean and covariance matrix
        mean = x[:, :c, :, :]
        cov = x[:, c:, :, :].reshape(-1, c, c, h, w)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean =  mean.to(dtype=dtype)
        rot = rot.to(dtype=dtype)
        rot = rot.permute(0, 3, 4, 1, 2)
        # # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(rot, centroid, 'p ... c1 c, b c h w  -> p c1 h w')
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)
    return _fitness, centroid, solution_length