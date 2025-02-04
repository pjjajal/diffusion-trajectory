from abc import ABC, abstractmethod

import torch
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline


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
        generator: torch.Generator = torch.Generator(0),
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
        generator: torch.Generator = torch.Generator(0),
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
        )

        # these embeddings will have
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
            device=self.device,
            dtype=self.pipeline.dtype,
            generator=self.generator,
        )
        return latents

    @torch.inference_mode()
    def __call__(self, noise_injection=None):
        # noise injection happens here
        latents = self.latents
        if noise_injection is not None:
            # this scales the noise injection to the initial noise sigma 
            # (this gets undone by the .prepare_latents() method of the pipeline)
            # this required otherwise the noise injection will be too large
            noise_injection = noise_injection / self.pipeline.scheduler.init_noise_sigma
            latents = self.latents + noise_injection
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
            output_type="np",
        )
        return images.images
