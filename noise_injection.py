import torch
from einops import einsum


# this function parameterizes the noise as the difference between the orginal latent and a rotated + translated version of it.
def rotational_transform(
    sample_fn,
    fitness_fn,
    latent_shape,
    device,
    center=None,
    mean_scale=1,
    dtype=torch.float32,
):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = c + c**2
    centroid = (
        center
        if center is not None
        else torch.zeros((b, c, h, w)).to(dtype=dtype, device=device)
    )

    def _inner_fn(x):
        x = x.reshape(-1, c + c**2)
        # slice out the mean and covariance matrix
        mean = x[:, :c]
        cov = x[:, c:].reshape(-1, c, c)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=dtype)

        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(centroid, rot, "b c h w, p c1 c -> p c1 h w")
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return samples

    def _fitness(x):
        samples = _inner_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)

    return _fitness, _inner_fn, centroid, solution_length


def svd_rot_transform(
    sample_fn,
    fitness_fn,
    latent_shape,
    device,
    center=None,
    mean_scale=1,
    bound=0.05,
    dtype=torch.float32,
):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = c + c**2
    centroid = (
        center
        if center is not None
        else torch.zeros((b, c, h, w)).to(dtype=dtype, device=device)
    )

    def _inner_fn(x):
        x = x.reshape(-1, c + c**2)
        # slice out the mean and covariance matrix
        mean = x[:, :c]
        cov = x[:, c:].reshape(-1, c, c)
        # svd to get the rotation matrix.
        u, s, v = torch.linalg.svd(cov)
        rot = u @ torch.diag_embed(s.clamp((1 - bound), (1 + bound))) @ v.mT
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=dtype)

        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(centroid, rot, "b c h w, p c1 c -> p c1 h w")
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return samples

    def _fitness(x):
        samples = _inner_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)

    return _fitness, _inner_fn, centroid, solution_length

def multi_axis_rotational_transform(
    sample_fn,
    fitness_fn,
    latent_shape,
    device,
    center=None,
    mean_scale=1,
    dtype=torch.float32,
):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = c + h + w + c**2 + h ** 2 + w ** 2
    centroid = (
        center
        if center is not None
        else torch.zeros((b, c, h, w)).to(dtype=dtype, device=device)
    )
    slice_length_c = c + c**2
    slice_length_h = h + h**2 + slice_length_c
    slice_length_w = w + w**2 + slice_length_h
    def _inner_fn(x):
        x = x.reshape(-1, solution_length)
        # slice out the mean and covariance matrix
        axis_c = x[:, :slice_length_c]
        axis_h = x[:, slice_length_c:slice_length_h]
        axis_w = x[:, slice_length_h:slice_length_w]

        # c-axis
        mean = axis_c[:, :c]
        cov = axis_c[:, c:].reshape(-1, c, c)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=dtype)
        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(centroid, rot, "b c h w, p c1 c -> p c1 h w")
        x = x + mean_scale * mean

        # h-axis
        mean = axis_h[:, :h]
        cov = axis_h[:, h:].reshape(-1, h, h)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(device, dtype=dtype).unsqueeze(1).unsqueeze(-1)
        rot = rot.to(device, dtype=dtype)
        x = einsum(x, rot, "p c h w, p h1 h -> p c h1 w")
        x = x + mean_scale * mean

        # w-axis
        mean = axis_w[:, :w]
        cov = axis_w[:, w:].reshape(-1, w, w)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(device, dtype=dtype).unsqueeze(1).unsqueeze(1)
        rot = rot.to(device, dtype=dtype)
        x = einsum(x, rot, "p c h w, p w1 w -> p c h w1")
        x = x + mean_scale * mean
        
        x = x - centroid
        samples = sample_fn(x)
        return samples

    def _fitness(x):
        samples = _inner_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)

    return _fitness, _inner_fn, centroid, solution_length


def multi_axis_svd_rot_transform(
    sample_fn,
    fitness_fn,
    latent_shape,
    device,
    center=None,
    mean_scale=1,
    bound=0.05,
    dtype=torch.float32,
):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = c + h + w + c**2 + h ** 2 + w ** 2
    centroid = (
        center
        if center is not None
        else torch.zeros((b, c, h, w)).to(dtype=dtype, device=device)
    )
    slice_length_c = c + c**2
    slice_length_h = h + h**2 + slice_length_c
    slice_length_w = w + w**2 + slice_length_h
    def _inner_fn(x):
        x = x.reshape(-1, solution_length)
        # slice out the mean and covariance matrix
        axis_c = x[:, :slice_length_c]
        axis_h = x[:, slice_length_c:slice_length_h]
        axis_w = x[:, slice_length_h:slice_length_w]

        # c-axis
        mean = axis_c[:, :c]
        cov = axis_c[:, c:].reshape(-1, c, c)
        # svd decomposition to get the rotation matrix.
        u, s, v = torch.linalg.svd(cov)
        rot = u @ torch.diag_embed(s.clamp((1 - bound), (1 + bound))) @ v.mT
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=dtype)
        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(centroid, rot, "b c h w, p c1 c -> p c1 h w")
        x = x + mean_scale * mean

        # h-axis
        mean = axis_h[:, :h]
        cov = axis_h[:, h:].reshape(-1, h, h)
        # svd decomposition to get the rotation matrix.
        u, s, v = torch.linalg.svd(cov)
        rot = u @ torch.diag_embed(s.clamp((1 - bound), (1 + bound))) @ v.mT
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(device, dtype=dtype).unsqueeze(1).unsqueeze(-1)
        rot = rot.to(device, dtype=dtype)
        x = einsum(x, rot, "p c h w, p h1 h -> p c h1 w")
        x = x + mean_scale * mean

        # w-axis
        mean = axis_w[:, :w]
        cov = axis_w[:, w:].reshape(-1, w, w)
        # svd decomposition to get the rotation matrix.
        u, s, v = torch.linalg.svd(cov)
        rot = u @ torch.diag_embed(s.clamp((1 - bound), (1 + bound))) @ v.mT
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(device, dtype=dtype).unsqueeze(1).unsqueeze(1)
        rot = rot.to(device, dtype=dtype)
        x = einsum(x, rot, "p c h w, p w1 w -> p c h w1")
        x = x + mean_scale * mean
        
        x = x - centroid
        samples = sample_fn(x)
        return samples

    def _fitness(x):
        samples = _inner_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)

    return _fitness, _inner_fn, centroid, solution_length


def rotational_transform_inject_multiple(
    sample_fn,
    fitness_fn,
    latent_shape,
    device,
    injection_steps,
    center=None,
    mean_scale=1,
    dtype=torch.float32,
):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = (c + c**2) * injection_steps
    centroid = (
        center[:, :injection_steps]
        if center is not None
        else torch.zeros((b, injection_steps, c, h, w)).to(dtype=dtype, device=device)
    )

    def _inner_fn(x):
        x = x.reshape(-1, injection_steps, c + c**2)
        # slice out the mean and covariance matrix
        mean = x[:, :, :c]
        cov = x[:, :, c:].reshape(-1, injection_steps, c, c)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)  # [b, injection_steps, c, c]
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = (
            mean.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        )  # [b, injection_steps, c, 1, 1]
        rot = rot.to(device, dtype=dtype)  # [b, injection_steps, c, c]

        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(centroid, rot, "b t c h w, p t c1 c -> p t c1 h w")
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return samples

    def _fitness(x):
        samples = _inner_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)

    return _fitness, _inner_fn, centroid, solution_length


# this is another version of the rotational transform, where we have two means and a covariance matrix.
# the centroid is shifted (mean_2), rotated, and shifted (mean) again.
# this seems to work okay, results are not as good as rotation + translation.
def rotational_transform_double_shift(
    sample_fn,
    fitness_fn,
    latent_shape,
    device,
    center=None,
    mean_scale=1,
    mean_scale_2=1,
    dtype=torch.float32,
):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = 2 * c + c**2
    centroid = (
        center
        if center is not None
        else torch.zeros((b, c, h, w)).to(dtype=dtype, device=device)
    )

    def _fitness(x):
        x = x.reshape(-1, 2 * c + c**2)
        # slice out the mean and covariance matrix
        mean = x[:, :c]
        mean_2 = x[:, c : 2 * c]
        cov = x[:, 2 * c :].reshape(-1, c, c)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        mean_2 = mean_2.to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        rot = rot.to(device, dtype=dtype)
        # rot = rot.permute(0, 3, 1, 2)
        # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(centroid + mean_scale_2 * mean_2, rot, "b c h w, p c1 c -> p c1 h w")
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)

    return _fitness, centroid, solution_length


# this one does not work at all, do not try to use it.
def rotational_transform_all(
    sample_fn,
    fitness_fn,
    latent_shape,
    device,
    center=None,
    mean_scale=1,
    dtype=torch.float32,
):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = (c + c**2) * h * w
    centroid = (
        center
        if center is not None
        else torch.zeros((b, c, h, w)).to(dtype=dtype, device=device)
    )

    def _fitness(x):
        x = x.reshape(-1, c + c**2, h, w)
        # slice out the mean and covariance matrix
        mean = x[:, :c, :, :]
        cov = x[:, c:, :, :].reshape(-1, c, c, h, w)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = mean.to(dtype=dtype)
        rot = rot.to(dtype=dtype)
        rot = rot.permute(0, 3, 4, 1, 2)
        # # rotate, translate and find the difference between the original and the rotated + translated version.
        x = einsum(rot, centroid, "p ... c1 c, b c h w  -> p c1 h w")
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return torch.cat([fitness_fn(sample.unsqueeze(0)) for sample in samples], dim=0)

    return _fitness, centroid, solution_length
