from abc import ABC, abstractmethod

import torch
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import PixArtAlphaPipeline
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import PixArtSigmaPipeline
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import (
    LatentConsistencyModelPipeline,
)
from einops import einsum


class SamplingPipeline(ABC):
    def __init__(
        self,
        pipeline: DiffusionPipeline,
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 7.0,
        height: int = 512,
        width: int = 512,
        generator: torch.Generator = torch.Generator(),
        output_type: str = "pil",
    ):
        super().__init__()
        self.device = pipeline.device
        self.pipeline = pipeline
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.generator = generator
        self.guidance_scale = guidance_scale
        self.classifier_free_guidance = classifier_free_guidance or guidance_scale > 0.0
        self.output_type = output_type

    @abstractmethod
    def embed_text(self, prompt: list[str] | None = None):
        raise NotImplementedError

    @abstractmethod
    def generate_latents(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, noise_injection=None):
        raise NotImplementedError


class SDXLSamplingPipeline(SamplingPipeline):
    def __init__(
        self,
        pipeline: StableDiffusionXLPipeline,
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 7,
        height: int = 512,
        width: int = 512,
        generator: torch.Generator = torch.Generator(),
        add_noise: bool = True,
        output_type: str = "pil",
    ):
        super().__init__(
            pipeline,
            prompt,
            num_inference_steps,
            classifier_free_guidance,
            guidance_scale,
            height,
            width,
            generator,
            output_type
        )
        self.add_noise = add_noise
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.embed_text(prompt)
        self.latents = self.generate_latents()

    @torch.inference_mode()
    def embed_text(self, prompt: str):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            do_classifier_free_guidance=self.classifier_free_guidance,
            num_images_per_prompt=1,
        )
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    @torch.inference_mode()
    def generate_latents(self):
        num_channel_latents = self.pipeline.unet.config.in_channels
        height = int(self.height) // self.pipeline.vae_scale_factor
        width = int(self.width) // self.pipeline.vae_scale_factor
        latents = torch.randn(
            (1, num_channel_latents, height, width),
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
            generator=self.generator,
        )
        return latents

    def regenerate_latents(self):
        self.latents = self.generate_latents()

    def rembed_text(self, prompt):
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.embed_text(prompt)

    @torch.inference_mode()
    def __call__(self, noise_injection=None):
        # noise injection happens here
        latents = self.latents
        if noise_injection is not None:
            # this scales the noise injection to the initial noise sigma
            # (this gets undone by the .prepare_latents() method of the pipeline)
            # this required otherwise the noise injection will be too large
            noise_injection = noise_injection / self.pipeline.scheduler.init_noise_sigma
            latents = latents + noise_injection if self.add_noise else noise_injection
        latents = latents.to(self.device, dtype=self.pipeline.dtype)
        images = self.pipeline(
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            prompt_embeds=self.prompt_embeds,
            negative_prompt_embeds=self.negative_prompt_embeds,
            pooled_prompt_embeds=self.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds,
            generator=self.generator,  # TODO this may need to be changed to be a seeded generator.
            num_images_per_prompt=latents.shape[0],
            latents=latents,
            output_type=self.output_type,
        )
        return images.images


class SD3SamplingPipeline(SamplingPipeline):
    def __init__(
        self,
        pipeline: StableDiffusion3Pipeline,
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 7,
        height: int = 512,
        width: int = 512,
        generator: torch.Generator = torch.Generator(),
        add_noise: bool = True,
        output_type: str = "pil",
    ):
        super().__init__(
            pipeline,
            prompt,
            num_inference_steps,
            classifier_free_guidance,
            guidance_scale,
            height,
            width,
            generator,
            output_type
        )
        self.add_noise = add_noise
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.embed_text(prompt)
        self.latents = self.generate_latents()

    @torch.inference_mode()
    def embed_text(self, prompt: str):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            device=self.device,
            do_classifier_free_guidance=self.classifier_free_guidance,
            num_images_per_prompt=1,
        )
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    @torch.inference_mode()
    def generate_latents(self):
        num_channel_latents = self.pipeline.transformer.config.in_channels
        height = int(self.height) // self.pipeline.vae_scale_factor
        width = int(self.width) // self.pipeline.vae_scale_factor
        latents = torch.randn(
            (1, num_channel_latents, height, width),
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
            generator=self.generator,
        )
        return latents

    def regenerate_latents(self):
        self.latents = self.generate_latents()

    def rembed_text(self, prompt):
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.embed_text(prompt)

    def update_latents(self, latent_update):
        if not self.static_latents:
            print("Updating latents")
            self.latents = latent_update(self.latents)

    @torch.inference_mode()
    def __call__(self, noise_injection=None):
        # noise injection happens here
        latents = self.latents
        if noise_injection is not None:
            latents = latents + noise_injection if self.add_noise else noise_injection
        latents = latents.to(self.device, dtype=self.pipeline.dtype)
        images = self.pipeline(
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            prompt_embeds=self.prompt_embeds.expand(latents.shape[0], -1, -1),
            negative_prompt_embeds=self.negative_prompt_embeds.expand(
                latents.shape[0], -1, -1
            ),
            pooled_prompt_embeds=self.pooled_prompt_embeds.expand(latents.shape[0], -1),
            negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds.expand(
                latents.shape[0], -1
            ),
            generator=self.generator,  # TODO this may need to be changed to be a seeded generator.
            latents=latents,
            output_type=self.output_type,
        )
        return images.images


class PixArtSigmaSamplingPipeline(SamplingPipeline):
    def __init__(
        self,
        pipeline: PixArtSigmaPipeline,
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 7,
        height: int = 512,
        width: int = 512,
        generator=torch.Generator(),
        add_noise: bool = True,
        output_type: str = "pil",
    ):
        super().__init__(
            pipeline,
            prompt,
            num_inference_steps,
            classifier_free_guidance,
            guidance_scale,
            height,
            width,
            generator,
            output_type
        )

        self.add_noise = add_noise
        (
            self.prompt_embeds,
            self.prompt_attention_mask,
            self.negative_prompt_embeds,
            self.negative_prompt_attention_mask,
        ) = self.embed_text(prompt)
        self.latents = self.generate_latents()

    @torch.inference_mode()
    def embed_text(self, prompt: str):
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            do_classifier_free_guidance=self.classifier_free_guidance,
            num_images_per_prompt=1,
        )
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    @torch.inference_mode()
    def generate_latents(self):
        num_channel_latents = self.pipeline.transformer.config.in_channels
        height = int(self.height) // self.pipeline.vae_scale_factor
        width = int(self.width) // self.pipeline.vae_scale_factor
        latents = torch.randn(
            (1, num_channel_latents, height, width),
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
            generator=self.generator,
        )
        return latents

    def regenerate_latents(self):
        self.latents = self.generate_latents()

    def rembed_text(self, prompt):
        (
            self.prompt_embeds,
            self.prompt_attention_mask,
            self.negative_prompt_embeds,
            self.negative_prompt_attention_mask,
        ) = self.embed_text(prompt)

    @torch.inference_mode()
    def __call__(self, noise_injection=None, noise_transform=None):
        # noise injection happens here
        latents = self.latents
        if noise_injection is not None:
            latents = latents + noise_injection if self.add_noise else noise_injection
        if noise_transform is not None:
            latents = noise_transform(latents)
        latents = latents.to(self.device, dtype=self.pipeline.dtype)
        images = self.pipeline(
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            prompt_embeds=self.prompt_embeds.expand(latents.shape[0], -1, -1),
            negative_prompt_embeds=self.negative_prompt_embeds.expand(
                latents.shape[0], -1, -1
            ),
            prompt_attention_mask=self.prompt_attention_mask.expand(
                latents.shape[0], -1
            ),
            negative_prompt_attention_mask=self.negative_prompt_attention_mask.expand(
                latents.shape[0], -1
            ),
            generator=self.generator,  # TODO this may need to be changed to be a seeded generator.
            latents=latents,
            output_type=self.output_type,
            prompt=None,
            negative_prompt=None,
        )
        return images.images


class PixArtAlphaSamplingPipeline(SamplingPipeline):
    def __init__(
        self,
        pipeline: PixArtAlphaPipeline,
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 7,
        height: int = 512,
        width: int = 512,
        generator=torch.Generator(),
        add_noise: bool = True,
        output_type: str = "pil",
    ):
        super().__init__(
            pipeline,
            prompt,
            num_inference_steps,
            classifier_free_guidance,
            guidance_scale,
            height,
            width,
            generator,
            output_type
        )

        self.add_noise = add_noise
        (
            self.prompt_embeds,
            self.prompt_attention_mask,
            self.negative_prompt_embeds,
            self.negative_prompt_attention_mask,
        ) = self.embed_text(prompt)
        self.latents = self.generate_latents()

    @torch.inference_mode()
    def embed_text(self, prompt: str):
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            do_classifier_free_guidance=self.classifier_free_guidance,
            num_images_per_prompt=1,
        )
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    @torch.inference_mode()
    def generate_latents(self):
        num_channel_latents = self.pipeline.transformer.config.in_channels
        height = int(self.height) // self.pipeline.vae_scale_factor
        width = int(self.width) // self.pipeline.vae_scale_factor
        latents = torch.randn(
            (1, num_channel_latents, height, width),
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
            generator=self.generator,
        )
        return latents

    def regenerate_latents(self):
        self.latents = self.generate_latents()

    def rembed_text(self, prompt):
        (
            self.prompt_embeds,
            self.prompt_attention_mask,
            self.negative_prompt_embeds,
            self.negative_prompt_attention_mask,
        ) = self.embed_text(prompt)

    @torch.inference_mode()
    def __call__(self, noise_injection=None, noise_transform=None):
        # noise injection happens here
        latents = self.latents
        if noise_injection is not None:
            latents = latents + noise_injection if self.add_noise else noise_injection
        if noise_transform is not None:
            latents = noise_transform(latents)
        latents = latents.to(self.device, dtype=self.pipeline.dtype)
        images = self.pipeline(
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            prompt_embeds=self.prompt_embeds.expand(latents.shape[0], -1, -1),
            negative_prompt_embeds=self.negative_prompt_embeds.expand(
                latents.shape[0], -1, -1
            ),
            prompt_attention_mask=self.prompt_attention_mask.expand(
                latents.shape[0], -1
            ),
            negative_prompt_attention_mask=self.negative_prompt_attention_mask.expand(
                latents.shape[0], -1
            ),
            generator=self.generator,  # TODO this may need to be changed to be a seeded generator.
            latents=latents,
            output_type=self.output_type,
            prompt=None,
            negative_prompt=None,
        )
        return images.images


class LCMSamplingPipeline(SamplingPipeline):
    def __init__(
        self,
        pipeline: LatentConsistencyModelPipeline,
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 7,
        height: int = 512,
        width: int = 512,
        generator=torch.Generator(),
        add_noise: bool = True,
        output_type: str = "pil",
    ):
        super().__init__(
            pipeline,
            prompt,
            num_inference_steps,
            classifier_free_guidance,
            guidance_scale,
            height,
            width,
            generator,
            output_type
        )

        self.add_noise = add_noise
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
        ) = self.embed_text(prompt)
        self.latents = self.generate_latents()

    @torch.inference_mode()
    def embed_text(self, prompt: str):
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            do_classifier_free_guidance=self.classifier_free_guidance,
            num_images_per_prompt=1,
        )
        return (
            prompt_embeds,
            negative_prompt_embeds,
        )

    @torch.inference_mode()
    def generate_latents(self):
        num_channel_latents = self.pipeline.unet.config.in_channels
        height = int(self.height) // self.pipeline.vae_scale_factor
        width = int(self.width) // self.pipeline.vae_scale_factor
        latents = torch.randn(
            (1, num_channel_latents, height, width),
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
            generator=self.generator,
        )
        return latents

    def regenerate_latents(self):
        self.latents = self.generate_latents()

    def rembed_text(self, prompt):
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
        ) = self.embed_text(prompt)

    @torch.inference_mode()
    def __call__(self, noise_injection=None, noise_transform=None):
        # noise injection happens here
        latents = self.latents
        if noise_injection is not None:
            latents = latents + noise_injection if self.add_noise else noise_injection
        if noise_transform is not None:
            latents = noise_transform(latents)
        latents = latents.to(self.device, dtype=self.pipeline.dtype)
        images = self.pipeline(
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            prompt_embeds=self.prompt_embeds.expand(latents.shape[0], -1, -1),
            generator=self.generator,  # TODO this may need to be changed to be a seeded generator.
            latents=latents,
            output_type=self.output_type,
            prompt=None,
        )
        return images.images
