import torch
from PIL import Image
from tqdm import tqdm
import os


# def embed_text(tokenizer, text_encoder, prompt: list[str]):
#     batch_size = len(prompt)
#     text_input = tokenizer(
#         prompt,
#         return_tensors="pt",
#         padding="max_length",
#         max_length=tokenizer.model_max_length,
#         truncation=True,
#     )
#     with torch.no_grad():
#         text_embeddings = text_encoder(text_input.input_ids.to(text_encoder.device))[0]

#     # unconditional text
#     max_length = text_input.input_ids.shape[-1]
#     uncond_input = tokenizer(
#         [""] * batch_size,
#         padding="max_length",
#         max_length=max_length,
#         return_tensors="pt",
#     )
#     uncond_embeddings = text_encoder(uncond_input.input_ids.to(text_encoder.device))[0]

#     text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
#     return text_embeddings

# def generate_latent(
#     vae_config, unet_config, generator, device, height=512, width=512, batch_size=1
# ):
#     compression_factor = 2 ** (len(vae_config.block_out_channels) - 1)
#     latent_height = height // compression_factor
#     latent_width = width // compression_factor
#     return torch.randn(
#         (batch_size, unet_config.in_channels, latent_height, latent_width),
#         generator=generator,
#         device=device,
#     )

# def sample_latents(
#     unet,
#     scheduler,
#     encoder_hidden_states,
#     latents,
#     num_inference_steps,
#     guidance_scale=0.5,
# ):
#     scheduler.set_timesteps(num_inference_steps)
#     latents = latents * scheduler.init_noise_sigma
#     latent_model_input = latents

#     for t in scheduler.timesteps:
#         latent_model_input = torch.cat([latents] * 2)
#         latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#         # predict the noise residual
#         with torch.no_grad():
#             noise_pred = unet(
#                 latent_model_input, t, encoder_hidden_states=encoder_hidden_states
#             ).sample

#         # perform guidance
#         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#         noise_pred = noise_pred_uncond + guidance_scale * (
#             noise_pred_text - noise_pred_uncond
#         )

#         # compute the previous noisy sample x_t -> x_t-1
#         latents = scheduler.step(noise_pred, t, latents).prev_sample

#     return latents

# # this is written in this way to expose a callable function which can be used with a black-box optimization algo.
# # it should be noted that sampling will be determinstic because the latents are fixed for the callable function.
# # to get a different sample, the latents should be re-sampled, i.e., initialize the generator differently.
# def diffusion_sample(
#     pipeline,
#     prompt: list[str],
#     num_inference_steps: int,
#     generator: torch.Generator,
#     guidance_scale: float = 0.5,
#     height=512,
#     width=512,
#     batch_size=1,
# ):
#     tokenizer = pipeline.tokenizer
#     text_encoder = pipeline.text_encoder
#     vae = pipeline.vae
#     unet = pipeline.unet
#     scheduler = pipeline.scheduler
#     dtype = pipeline.dtype

#     # print(f"using dtype: {dtype}")

#     if len(prompt) != batch_size:
#         prompt = prompt * batch_size
#     text_embeddings = embed_text(tokenizer, text_encoder, prompt).to(dtype=dtype)
#     latents = generate_latent(
#         vae.config,
#         unet.config,
#         generator,
#         device=vae.device,
#         height=height,
#         width=width,
#         batch_size=batch_size,
#     ).to(dtype=dtype)

#     def _step(x, t, i, noise_injection=None):
#         b = x.shape[0]
#         latent_model_input = torch.cat([x] * 2)
#         latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#         # predict the noise residual
#         encoder_hidden_states = text_embeddings
#         if b > 1:
#             encoder_hidden_states = encoder_hidden_states.repeat_interleave(b, dim=0)
#         with torch.no_grad():
#             noise_pred = unet(
#                 latent_model_input, t, encoder_hidden_states=encoder_hidden_states
#             ).sample

#         # perform guidance
#         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#         noise_pred = noise_pred_uncond + guidance_scale * (
#             noise_pred_text - noise_pred_uncond
#         )

#         # add noise injection
#         if noise_injection is not None:
#             noise_pred = noise_pred + noise_injection

#         # compute the previous noisy sample x_t -> x_t-1
#         x = scheduler.step(noise_pred, t, x).prev_sample
#         return x

#     def sample(noise_injection=None):
#         _latents = latents * scheduler.init_noise_sigma + (
#             noise_injection if noise_injection is not None else 0.0
#         )
#         scheduler.set_timesteps(num_inference_steps)

#         progress_bar = enumerate(scheduler.timesteps)

#         for i, t in progress_bar:
#             _latents = _step(_latents, t, i + 1, None)

#         return decode_latents(_latents, vae)

#     return (
#         sample,
#         latents,
#         num_inference_steps + scheduler.config.steps_offset + 1,
#         dtype,
#     )


# def decode_latents(latents, vae):
#     latents = 1 / vae.config.scaling_factor * latents
#     with torch.no_grad():
#         samples = vae.decode(latents).sample
#     return samples


class DiffusionSample:
    def __init__(
        self,
        pipeline,
        prompt: list[str],
        num_inference_steps: int,
        generator: torch.Generator,
        guidance_scale: float = 0.5,
        height=512,
        width=512,
        batch_size=1,
        inject_multiple=False,  # denotes whether to inject noise at multiple denoising steps.
        inject_multiple_noise_scale: list[float] | float = 1.0,
    ):
        # injection flag
        self.inject_multiple = inject_multiple
        self.inject_multiple_noise_scale = (
            inject_multiple_noise_scale
            if isinstance(inject_multiple_noise_scale, list)
            else [inject_multiple_noise_scale]
        )

        # torch.Generator is a device-agnostic random number generator.
        self.generator = generator

        # pipeline components
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder
        self.vae = pipeline.vae
        self.unet = pipeline.unet
        self.scheduler = pipeline.scheduler
        self.dtype = pipeline.dtype

        # print(f"using dtype: {self.dtype}")

        # pipeline configuration
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale

        self.embed_text()
        self.generate_latents()

    def __call__(self, noise_injection=None):
        if self.inject_multiple and noise_injection is not None:
            return self.sample_inject_multiple(noise_injection)
        return self.sample(noise_injection)

    # this is the default behavior, i.e., regular diffusion sampling and noise injection at t=T (i.e., at the latent).
    def sample(self, noise_injection=None):
        _latents = self.latents * self.scheduler.init_noise_sigma + (
            noise_injection if noise_injection is not None else 0.0
        )  # noise inject only at t=T latent, not at intermediate steps.
        self.scheduler.set_timesteps(self.num_inference_steps)
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            _latents = self._step(
                _latents, t, None
            )  # note that noise is not injected here.
        return self.decode_latents(_latents)

    def sample_inject_multiple(self, noise_injection):
        # noise injection at multiple denoising steps.
        # noise_injection is a tensor of shape [b, t, c, h, w], where t the noise for the (T-t)-th denoising step.
        num_noise = noise_injection.shape[1]
        noise_scale = self._get_inject_multiple_noise_scale(0, num_noise)
        _latents = self.latents * self.scheduler.init_noise_sigma + (
            noise_scale * noise_injection[:, 0] if noise_injection is not None else 0.0
        )  # noise inject only at t=T latent, not at intermediate steps.
        self.scheduler.set_timesteps(self.num_inference_steps)
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # noise_weight is the scaling factor for the noise at the i-th denoising step.
            # this makes it so that the preturbations are scaled down as t->0.
            noise_weight = (1 - self.scheduler.alphas_cumprod[t]) ** 0.5
            noise_scale = self._get_inject_multiple_noise_scale(i, num_noise)
            noise_weight = noise_weight * noise_scale
            _latents = self._step(
                _latents,
                t,
                noise_weight * noise_injection[:, i] if i < num_noise and i != 0 else None,
            )  # note that noise is injected here at intermediate steps.
        return self.decode_latents(_latents)

    def collect_model_preds(self):
        _latents = self.latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.num_inference_steps)
        preds = _latents
        preds_list = []
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            preds_list.append(preds)
            _latents, preds = self._step(_latents, t, None, return_pred=True)
        return torch.stack(preds_list, dim=1)  # [b, t, c, h, w]

    def noise_injection_args(self):
        centers = (
            self.latents if not self.inject_multiple else self.collect_model_preds()
        )
        return (
            centers,  # when inject_multiple is False, centers is the latent tensor, otherwise it is the latent + model_preds (post_guidance) tensor.
            self.num_inference_steps + self.scheduler.config.steps_offset,
            self.dtype,
        )

    def embed_text(self, prompt: list[str] | None = None):
        prompt = prompt or self.prompt
        self.prompt = prompt
        if len(prompt) != self.batch_size:
            prompt = prompt = prompt * self.batch_size
        self.text_embeddings = self._embed_text(
            self.tokenizer, self.text_encoder, prompt
        ).to(dtype=self.dtype)
        return self.text_embeddings

    def generate_latents(self):
        self.latents = self._generate_latents(
            self.vae.config,
            self.unet.config,
            self.generator,
            device=self.vae.device,
            height=self.height,
            width=self.width,
            batch_size=self.batch_size,
        ).to(dtype=self.dtype)
        return self.latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        with torch.no_grad():
            samples = self.vae.decode(latents).sample
        return samples

    def _get_inject_multiple_noise_scale(self, i, num_noise):
        if i < len(
            self.inject_multiple_noise_scale
        ):
            return self.inject_multiple_noise_scale[i]
        return self.inject_multiple_noise_scale[-1]

    def _embed_text(self, tokenizer, text_encoder, prompt: list[str]):
        batch_size = len(prompt)
        text_input = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        with torch.no_grad():
            text_embeddings = text_encoder(
                text_input.input_ids.to(text_encoder.device)
            )[0]

        # unconditional text
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(text_encoder.device)
        )[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def _generate_latents(
        self,
        vae_config,
        unet_config,
        generator,
        device,
        height=512,
        width=512,
        batch_size=1,
    ):
        compression_factor = 2 ** (len(vae_config.block_out_channels) - 1)
        latent_height = height // compression_factor
        latent_width = width // compression_factor
        return torch.randn(
            (batch_size, unet_config.in_channels, latent_height, latent_width),
            generator=generator,
            device=device,
        )

    def _step(self, x, t, noise_injection=None, return_pred=False):
        b = x.shape[0]
        latent_model_input = torch.cat([x] * 2)
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, timestep=t
        )

        # predict the noise residual
        encoder_hidden_states = self.text_embeddings
        if b > 1:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(b, dim=0)
        with torch.no_grad():
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=encoder_hidden_states
            ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # noise injection in the model prediction.
        if noise_injection is not None:
            noise_pred = noise_pred + noise_injection

        # compute the previous noisy sample x_t -> x_t-1
        x = self.scheduler.step(noise_pred, t, x).prev_sample
        return (x, noise_pred) if return_pred else x
