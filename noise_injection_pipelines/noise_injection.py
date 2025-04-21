import torch
from einops import einsum
from .diffusion_pt import DiffusionSample
from collections import deque
import numpy as np


# this function parameterizes the noise as the difference between the orginal latent and a rotated + translated version of it.
def rotational_transform(
    sample_fn,
    fitness_fn,
    latent_shape,
    center=None,
    mean_scale=1,
    dtype=torch.float32,
):
    if len(latent_shape) == 4:
        b, c, h, w = latent_shape
        solution_length = c * h * w
        # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
        # thus we must apply the transformation to the channel dimension of the latent space.
        solution_length = c + c**2
        centroid = (
            center
            if center is not None
            else torch.zeros((b, c, h, w)).to(dtype=dtype, device=sample_fn.device)
        )
    elif len(latent_shape) == 3:
        b, l, c = latent_shape
        solution_length = l * c
        solution_length = c + c**2
        centroid = (
            center
            if center is not None
            else torch.zeros((b, l, c)).to(dtype=dtype, device=sample_fn.device)
        )

    def _inner_fn(x):
        device = centroid.device
        x = x.reshape(-1, c + c**2)
        # slice out the mean and covariance matrix
        mean = x[:, :c]
        cov = x[:, c:].reshape(-1, c, c)
        # QR decomposition to get the rotation matrix.
        rot, scaling = torch.linalg.qr(cov)
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        if len(latent_shape) == 3:
            mean = torch.Tensor(mean).to(device, dtype=dtype).unsqueeze(1)
        else:
            mean = (
                torch.Tensor(mean).to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
            )
        rot = torch.Tensor(rot).to(device, dtype=dtype)

        # rotate, translate and find the difference between the original and the rotated + translated version.
        if len(latent_shape) == 4:
            x = einsum(centroid, rot, "b c h w, p c1 c -> p c1 h w")
        elif len(latent_shape) == 3:
            x = einsum(centroid, rot, "b l c, p c1 c -> p l c1")
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return samples

    def _fitness(x):
        samples = _inner_fn(x)
        return torch.cat(
            [
                (
                    fitness_fn(sample.unsqueeze(0))
                    if isinstance(sample_fn, DiffusionSample)
                    else fitness_fn(sample)
                )
                for sample in samples
            ],
            dim=0,
        )

    return _fitness, _inner_fn, centroid, solution_length


def svd_rot_transform(
    sample_fn,
    fitness_fn,
    latent_shape,
    center=None,
    mean_scale=1,
    bound=0.05,
    dtype=torch.float32,
):
    if len(latent_shape) == 4:
        b, c, h, w = latent_shape
        solution_length = c * h * w
        # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
        # thus we must apply the transformation to the channel dimension of the latent space.
        solution_length = c + c**2
        centroid = (
            center
            if center is not None
            else torch.zeros((b, c, h, w)).to(dtype=dtype, device=sample_fn.device)
        )
    elif len(latent_shape) == 3:
        b, l, c = latent_shape
        solution_length = l * c
        solution_length = c + c**2
        centroid = (
            center
            if center is not None
            else torch.zeros((b, l, c)).to(dtype=dtype, device=sample_fn.device)
        )

    def _inner_fn(x):
        device = centroid.device
        x = x.reshape(-1, c + c**2)
        # slice out the mean and covariance matrix
        mean = x[:, :c]
        cov = x[:, c:].reshape(-1, c, c)
        # svd to get the rotation matrix.
        u, s, v = torch.linalg.svd(cov)
        rot = u @ torch.diag_embed(s.clamp((1 - bound), (1 + bound))) @ v.mT
        # unsqueeze so that we can broadcast the mean and rotation matrix to the correct shape.
        mean = torch.Tensor(mean).to(device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
        rot = torch.Tensor(rot).to(device, dtype=dtype)

        # rotate, translate and find the difference between the original and the rotated + translated version.
        if len(latent_shape) == 4:
            x = einsum(centroid, rot, "b c h w, p c1 c -> p c1 h w")
        elif len(latent_shape) == 3:
            x = einsum(centroid, rot, "b l c, p c1 c -> p l c1")
        x = x + mean_scale * mean - centroid
        samples = sample_fn(x)
        return samples

    def _fitness(x):
        samples = _inner_fn(x)
        return torch.cat([fitness_fn(sample) for sample in samples], dim=0)

    return _fitness, _inner_fn, centroid, solution_length


def multi_axis_rotational_transform(
    sample_fn,
    fitness_fn,
    latent_shape,
    center=None,
    mean_scale=1,
    dtype=torch.float32,
):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = c + h + w + c**2 + h**2 + w**2
    centroid = (
        center
        if center is not None
        else torch.zeros((b, c, h, w)).to(dtype=dtype, device=sample_fn.device)
    )
    slice_length_c = c + c**2
    slice_length_h = h + h**2 + slice_length_c
    slice_length_w = w + w**2 + slice_length_h

    def _inner_fn(x):
        device = centroid.device
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
        return torch.cat([fitness_fn(sample) for sample in samples], dim=0)

    return _fitness, _inner_fn, centroid, solution_length


def multi_axis_svd_rot_transform(
    sample_fn,
    fitness_fn,
    latent_shape,
    center=None,
    mean_scale=1,
    bound=0.05,
    dtype=torch.float32,
):
    b, c, h, w = latent_shape
    # this is equal to the mean and a covariance matrix, that is [c] and [c, c] respectively.
    # thus we must apply the transformation to the channel dimension of the latent space.
    solution_length = c + h + w + c**2 + h**2 + w**2
    centroid = (
        center
        if center is not None
        else torch.zeros((b, c, h, w)).to(dtype=dtype, device=sample_fn.device)
    )
    slice_length_c = c + c**2
    slice_length_h = h + h**2 + slice_length_c
    slice_length_w = w + w**2 + slice_length_h

    def _inner_fn(x):
        device = centroid.device
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
        return torch.cat([fitness_fn(sample) for sample in samples], dim=0)

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


def noise(
    sample_fn,
    fitness_fn,
    latent_shape,
    device,
    dtype=torch.float32,
):
    if len(latent_shape) == 4:
        b, c, h, w = latent_shape
        solution_length = c * h * w
    elif len(latent_shape) == 3:
        b, l, d = latent_shape
        solution_length = l * d

    def _inner_fn(x):
        if len(latent_shape) == 4:
            x = x.reshape(-1, c, h, w).to(device, dtype=dtype)
        elif len(latent_shape) == 3:
            x = x.reshape(-1, l, d).to(device, dtype=dtype)
        samples = sample_fn(noise_injection=x)
        return samples

    def _fitness(x):
        samples = _inner_fn(x)
        return torch.cat(
            [
                (
                    fitness_fn(sample.unsqueeze(0))
                    if isinstance(sample_fn, DiffusionSample)
                    else fitness_fn(sample)
                )
                for sample in samples
            ],
            dim=0,
        )

    return _fitness, _inner_fn, sample_fn.latents, solution_length


class StatefulNoise:
    def __init__(
        self, sample_fn, fitness_fn, latent_shape, device, dtype=torch.float32
    ):
        super().__init__()
        self.sample_fn = sample_fn
        self.fitness_fn = fitness_fn
        self.latent_shape = latent_shape
        self.device = device
        self.dtype = dtype

        b, c, h, w = latent_shape

        self.c, self.h, self.w = c, h, w
        self.solution_length = c * h * w

        self.img_queue = deque()
        self.fitness_queue = deque()

    def _inner_fn(self, x):
        x = x.reshape(-1, self.c, self.h, self.w).to(self.device, dtype=self.dtype)
        samples = self.sample_fn(noise_injection=x)
        return samples

    def __call__(self, x):
        samples = self._inner_fn(x)

        fitness = [
            (
                self.fitness_fn(sample.unsqueeze(0))
                if isinstance(self.sample_fn, DiffusionSample)
                else self.fitness_fn(sample)
            )
            for sample in samples
        ]
        self.img_queue.extend(samples)
        self.fitness_queue.extend(fitness)
        return torch.cat(fitness, dim=0)

    def get_best_img(self):
        argmax = self.fitness_queue.index(max(self.fitness_queue))
        return self.img_queue[argmax]

    def get_best_fitness(self):
        return max(self.fitness_queue)
