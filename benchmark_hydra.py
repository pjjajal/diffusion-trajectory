import argparse
import logging
import random
import subprocess
import time
import warnings
from pathlib import Path, PosixPath, WindowsPath
from typing import *

import hydra
import numpy as np
import pandas
import torch
import wandb
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    UNet2DConditionModel,
    SD3Transformer2DModel,
    # BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
)
from diffusers.utils import export_to_gif, numpy_to_pil
from einops import einsum
from evotorch.algorithms import CEM, CMAES, SNES
from omegaconf import DictConfig
from PIL.Image import Image
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import eval_datasets
from evotorch.logging import StdOutLogger
from evo.vectorized_problem import VectorizedProblem
from fitness.fitness_fn import (
    compose_fitness_fns,
    clip_fitness_fn,
    aesthetic_fitness_fn,
    pickscore_fitness_fn,
    imagereward_fitness_fn,
    hpsv2_fitness_fn,
    brightness,
    relative_luminance,
    Novelty,
)
from noise_injection_pipelines.diffusion_pt import DiffusionSample
from noise_injection_pipelines.noise_injection import (
    rotational_transform,
    svd_rot_transform,
    noise,
)
from noise_injection_pipelines.sampling_pipelines import (
    SD3SamplingPipeline,
    SDXLSamplingPipeline,
)

warnings.filterwarnings("ignore")


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# this is used to save the config file to wandb
def flatten_dict(d: DictConfig):
    out = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            for k2, v2 in flatten_dict(v).items():
                out[k + "." + k2] = v2
        else:
            out[k] = v
    return out


def create_pipeline(pipeline_cfg: DictConfig):
    if pipeline_cfg.type == "sdxl":
        unet = None
        # if pipeline_cfg.quantize:
        #     load_in_4bit = pipeline_cfg.quantize_cfg.bits == "4bit"
        #     load_in_8bit = pipeline_cfg.quantize_cfg.bits == "8bit"
        #     quant_config = DiffusersBitsAndBytesConfig(
        #         load_in_4bit=load_in_4bit,
        #         load_in_8bit=load_in_8bit,
        #     )
        #     unet = UNet2DConditionModel.from_pretrained(
        #         pipeline_cfg.model_id,
        #         subfolder="unet",
        #         torch_dtype=DTYPE_MAP[pipeline_cfg.dtype],
        #         quant_config=quant_config,
        #         device_map=pipeline_cfg.device_map,
        #         cache_dir=pipeline_cfg.cache_dir,
        #         use_safetensors=True,
        #     )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pipeline_cfg.model_id,
            device_map=pipeline_cfg.device_map,
            torch_dtype=DTYPE_MAP[pipeline_cfg.dtype],
            cache_dir=pipeline_cfg.cache_dir,
            use_safetensors=True,
        ).to(pipeline_cfg.device)
    elif pipeline_cfg.type == "sd3":
        transformer = None
        # if pipeline_cfg.quantize:
        #     load_in_4bit = pipeline_cfg.quantize_cfg.bits == "4bit"
        #     load_in_8bit = pipeline_cfg.quantize_cfg.bits == "8bit"
        #     quant_config = DiffusersBitsAndBytesConfig(
        #         load_in_4bit=load_in_4bit,
        #         load_in_8bit=load_in_8bit,
        #     )
        #     transformer = SD3Transformer2DModel.from_pretrained(
        #         pipeline_cfg.model_id,
        #         subfolder="transformer",
        #         quantization_config=quant_config,
        #         torch_dtype=torch.float16,
        #         cache_dir=pipeline_cfg.cache_dir,
        #         use_safetensors=True,
        #     )
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            pipeline_cfg.model_id,
            device_map=pipeline_cfg.device_map,
            torch_dtype=DTYPE_MAP[pipeline_cfg.dtype],
            cache_dir=pipeline_cfg.cache_dir,
            use_safetensors=True,
        ).to(pipeline_cfg.device)
    return pipeline


def create_sampler(
    cfg: DictConfig,
    pipeline: DiffusionPipeline,
    generator: torch.Generator,
):
    if cfg.pipeline.type == "sdxl":
        return SDXLSamplingPipeline(
            pipeline=pipeline,
            prompt="",
            num_inference_steps=cfg.pipeline.num_inference_steps,
            classifier_free_guidance=cfg.pipeline.classifier_free_guidance,
            guidance_scale=cfg.pipeline.guidance_scale,
            generator=generator,
            add_noise=cfg.noise_injection.add_noise,
        )
    elif cfg.pipeline.type == "sd3":
        return SD3SamplingPipeline(
            pipeline=pipeline,
            prompt="",
            num_inference_steps=cfg.pipeline.num_inference_steps,
            classifier_free_guidance=cfg.pipeline.classifier_free_guidance,
            guidance_scale=cfg.pipeline.guidance_scale,
            generator=generator,
            add_noise=cfg.noise_injection.add_noise,
        )


def create_fitness_fn(cfg: DictConfig, prompt: str):
    fitness_cfg = cfg.fitness
    cache_dir = fitness_cfg.cache_dir
    fitness_fns = []
    weights = []
    if fitness_cfg.fns.brightness.active:
        fitness_fns.append(brightness)
        weights.append(fitness_cfg.fns.brightness.weight)
    if fitness_cfg.fns.clip.active:
        clip_prompt = prompt or fitness_cfg.fns.clip.prompt
        fitness_fns.append(
            clip_fitness_fn(
                clip_model_name=fitness_cfg.fns.clip.model_name,
                prompt=clip_prompt,
                cache_dir=cache_dir,
            )
        )
        weights.append(fitness_cfg.fns.clip.weight)
    if fitness_cfg.fns.aesthetic.active:
        fitness_fns.append(aesthetic_fitness_fn(cache_dir=cache_dir))
        weights.append(fitness_cfg.fns.aesthetic.weight)
    if fitness_cfg.fns.pick.active:
        pickscore_prompt = prompt or fitness_cfg.fns.pick.prompt
        fitness_fns.append(
            pickscore_fitness_fn(prompt=pickscore_prompt, cache_dir=cache_dir)
        )
        weights.append(fitness_cfg.fns.pick.weight)
    if fitness_cfg.fns.novelty.active:
        fitness_fns.append(
            Novelty(
                model_name=fitness_cfg.fns.novelty.model_name,
                top_k=fitness_cfg.fns.novelty.top_k,
                cache_dir=cache_dir,
            )
        )
        weights.append(fitness_cfg.fns.novelty.weight)
    if fitness_cfg.fns.hpsv2.active:
        hpsv2_prompt = prompt or fitness_cfg.fns.hpsv2.prompt
        fitness_fns.append(hpsv2_fitness_fn(prompt=hpsv2_prompt, cache_dir=cache_dir))
        weights.append(fitness_cfg.fns.hpsv2.weight)
    if fitness_cfg.fns.imagereward.active:
        imagereward_prompt = prompt or fitness_cfg.fns.imagereward.prompt
        fitness_fns.append(imagereward_fitness_fn(prompt=imagereward_prompt))
    return compose_fitness_fns(fitness_fns, weights)


def create_obj_fn(sample_fn, fitness_fn, cfg: DictConfig):
    if cfg.noise_injection.type == "rotational_transform":
        obj_fn, inner_fn, centroid, solution_length = rotational_transform(
            sample_fn=sample_fn,
            fitness_fn=fitness_fn,
            latent_shape=sample_fn.latents.shape,
            center=sample_fn.latents,
            mean_scale=cfg.noise_injection.mean_scale,
            dtype=sample_fn.pipeline.dtype,
        )
    elif cfg.noise_injection.type == "svd_rot_transform":
        obj_fn, inner_fn, centroid, solution_length = svd_rot_transform(
            sample_fn=sample_fn,
            fitness_fn=fitness_fn,
            latent_shape=sample_fn.latents.shape,
            center=sample_fn.latents,
            mean_scale=cfg.noise_injection.mean_scale,
            bound=cfg.noise_injection.bound,
            dtype=sample_fn.pipeline.dtype,
        )
    elif cfg.noise_injection_type == "noise":
        obj_fn, inner_fn, centroid, solution_length = noise(
            sample_fn=sample_fn,
            fitness_fn=fitness_fn,
            latent_shape=sample_fn.latents.shape,
            device=sample_fn.pipeline.device,
            dtype=sample_fn.pipeline.dtype,
        )
    else:
        raise NotImplementedError
    return obj_fn, inner_fn, centroid, solution_length


def create_solver(problem, latents, solver_cfg: DictConfig):
    if solver_cfg.algorithm == "cmaes":
        center_init = (
            latents.flatten() if solver_cfg.cmaes.center_init == "latents" else None
        )
        return CMAES(
            problem=problem,
            stdev_init=solver_cfg.cmaes.stdev_init,
            csa_squared=solver_cfg.cmaes.csa_squared,
            separable=solver_cfg.cmaes.separable,
            center_init=center_init,
        )
    elif solver_cfg.algorithm == "snes":
        center_init = (
            latents.flatten() if solver_cfg.snes.center_init == "latents" else None
        )
        return SNES(
            problem=problem,
            stdev_init=solver_cfg.snes.stdev_init,
            center_init=center_init,
        )
    
def measure_torch_device_memory_used_mb(device: torch.device) -> float:
	if device.type == "cuda":
		free, total = torch.cuda.mem_get_info(device)
		return (total - free) / 1024 ** 2
	else:
		return -1.0
    
def wandb_log(solver, step, img, prompt, running_time, device):
    wandb.log({
        "step": step,
        "pop_best_eval": solver.status["pop_best_eval"],
        "mean_eval": solver.status["mean_eval"],
        "median_eval": solver.status["median_eval"],
        "best_img": wandb.Image(img),
        "prompt": prompt,
        "running_time": running_time,
        "memory" : measure_torch_device_memory_used_mb(device)
    })


@torch.inference_mode()
def benchmark(benchmark_cfg: DictConfig, solver, sample_fn, inner_fn, prompt):
    start_time = time.time()
    step = 0

    frames = []
    # initial image
    img = numpy_to_pil(sample_fn())[0]
    if benchmark_cfg.wandb.active:
        wandb.log({
            "step": step,
            "best_img": wandb.Image(img),
            "prompt": prompt
        })
    frames.append(img)

    # run benchmark
    while True:
        step += 1
        solver.step()

        pop_best_sol = solver.status['pop_best'].values
        img = numpy_to_pil(inner_fn(pop_best_sol))[0]
        frames.append(img)

        running_time = time.time() - start_time
        if benchmark_cfg.wandb.active:
            wandb_log(solver, step, frames[-1], prompt, running_time, sample_fn.device)

        if benchmark_cfg.type == "till_time":
            if running_time >= benchmark_cfg.till_time:
                break
        elif benchmark_cfg.type == "till_steps":
            if step >= benchmark_cfg.for_steps:
                break
        elif benchmark_cfg.type == "till_reward":
            if solver.best_eval > benchmark_cfg.till_reward:
                break
    
    wandb_log(solver, step, frames[-1], prompt, running_time, sample_fn.device)

@hydra.main(config_path="configs")
def main(cfg: DictConfig):
    # set seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # set up wandb
    if cfg.benchmark.wandb.active:
        wandb.init(
            project=cfg.benchmark.wandb.project,
            config=flatten_dict(cfg),
        )

    # load eval dataset
    dataset = eval_datasets.create_dataset(cfg.dataset)

    # create pipeline
    pipeline = create_pipeline(cfg.pipeline)
    if cfg.pipeline.xformers_mem_eff_attn:
        pipeline.enable_xformers_memory_efficient_attention()
    if cfg.pipeline.cpu_offload:
        pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device=pipeline.device).manual_seed(cfg.seed)
    sample_fn = create_sampler(
        cfg,
        pipeline,
        generator=generator,
    )

    for x in dataset.iter(batch_size=1):
        sample_fn.regenerate_latents()
        sample_fn.rembed_text(x["prompt"])
        fitness_fn = create_fitness_fn(cfg, x["prompt"])
        obj_fn, inner_fn, centroid, solution_length = create_obj_fn(
            sample_fn, fitness_fn, cfg
        )
        problem = VectorizedProblem(
            cfg.fitness.objective,
            objective_func=obj_fn,
            dtype=np.dtype("float32"),
            splits=cfg.fitness.problem_splits,
            solution_length=solution_length,
            device=pipeline.device,
            initial_bounds=cfg.solver.initial_bounds
        )
        solver = create_solver(problem, centroid, cfg.solver)

        logger = StdOutLogger(solver)
        # run
        benchmark(cfg.benchmark, solver, sample_fn, inner_fn, x['prompt'])


if __name__ == "__main__":
    main()
