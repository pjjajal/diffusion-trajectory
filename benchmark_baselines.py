import os
import time
from functools import partial

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import random

import eval_datasets

from benchmark_hydra import (
    DTYPE_MAP,
    create_fitness_fn,
    create_pipeline,
    create_sampler,
    flatten_dict,
)
from tqdm import tqdm


def measure_torch_device_memory_used_mb(device: torch.device) -> float:
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return (total - free) / 1024**2
    else:
        return -1.0


def wandb_log(
    step, img, best_fitness, mean_fitness, median_fitness, prompt, running_time, device
):
    wandb.log(
        {
            "step": step,
            "pop_best_eval": best_fitness,
            "mean_eval": mean_fitness,
            "median_eval": median_fitness,
            "best_img": wandb.Image(img),
            "prompt": prompt[0],
            "running_time": running_time,
            "memory": measure_torch_device_memory_used_mb(device),
        }
    )


class ZeroOrderSearch:
    def __init__(self, fitness_fn, population, split_size, threshold, latents, objective):
        super().__init__()
        self.fitness_fn = fitness_fn
        self.population = population
        self.split_size = split_size
        self.threshold = threshold
        self.latents = latents
        self.objective = objective

    # Borrowed from https://github.com/sayakpaul/tt-scale-flux/blob/main/utils.py#L255
    def generate_neighbors(self, x, threshold=0.95, num_neighbors=4):
        """Courtesy: Willis Ma"""
        rng = np.random.Generator(np.random.PCG64())
        x_f = x.flatten(1)
        x_norm = torch.linalg.norm(
            x_f, dim=-1, keepdim=True, dtype=torch.float64
        ).unsqueeze(-2)
        u = x_f.unsqueeze(-2) / x_norm.clamp_min(1e-12)
        v = torch.from_numpy(
            rng.standard_normal(
                size=(num_neighbors, u.shape[-1]), dtype=np.float64
            )
        ).to(u.device)
        w = F.normalize(v - (v @ u.transpose(-2, -1)) * u, dim=-1)
        return (
            (x_norm * (threshold * u + np.sqrt(1 - threshold**2) * w))
            .reshape(num_neighbors, *x.shape[1:])
            .to(x.dtype)
        )
    
    def __call__(self, step, sample_fn):
        print(self.latents.shape)
        neighbours = self.generate_neighbors(
            self.latents,
            threshold=self.threshold,
            num_neighbors=self.population,
        )
        latents = torch.cat([self.latents, neighbours], dim=0)

        fitnesses = []
        for latent_split in latents.split(self.split_size):
            imgs = sample_fn(latent_split)
            fitnesses.extend([self.fitness_fn(img) for img in imgs])
        
        base_fitness = fitnesses[0]
        new_fitnesses = fitnesses[1:]
        if self.objective == "max":
            best_fitness = np.max(new_fitnesses)
            if best_fitness > base_fitness.item():
                print("Pivoting Latent, best fitness: ", best_fitness, "base fitness: ", base_fitness)
                best_idx = fitnesses.index(best_fitness.item())
                best_latent = latents[best_idx]
                self.latents = best_latent.unsqueeze(0)
        elif self.objective == "min":
            best_fitness = np.min(new_fitnesses)
            if best_fitness < base_fitness.item():
                print("Pivoting Latent, best fitness: ", best_fitness, "base fitness: ", base_fitness)
                best_idx = fitnesses.index(best_fitness.item())
                best_latent = latents[best_idx]
                self.latents = best_latent.unsqueeze(0)
        print("Best fitness: ", best_fitness, "base fitness: ", base_fitness)
        print(
            f"Step {step}: best fitness {best_fitness:.4f}, mean fitness {np.mean(fitnesses):.4f}, median fitness {np.median(fitnesses):.4f}"
        )
        return latents, fitnesses




def random_search(step, sample_fn, fitness_fn, population, split_size, objective):
    latents = torch.randn(
        (population, *sample_fn.latents.shape[1:]),
        device=sample_fn.device
    )
    # latents = torch.cat([sample_fn.generate_latents() for _ in range(population)])
    # latents = latents.to(sample_fn.device)

    fitnesses = []
    for latent_split in latents.split(split_size):
        imgs = sample_fn(latent_split)
        fitnesses.extend([fitness_fn(img) for img in imgs])

    if objective == "max":
        print(
            f"Step {step}: best fitness {np.max(fitnesses):.4f}, mean fitness {np.mean(fitnesses):.4f}, median fitness {np.median(fitnesses):.4f}"
        )
    elif objective == "min":
        print(
            f"Step {step}: best fitness {np.min(fitnesses):.4f}, mean fitness {np.mean(fitnesses):.4f}, median fitness {np.median(fitnesses):.4f}"
        )
    return latents, fitnesses


@torch.inference_mode()
def benchmark(benchmark_cfg: DictConfig, sample_fn, searcher, prompt, objective):
    start_time = time.time()
    step = 0

    # run benchmark
    while True:
        step += 1
        # generate images
        latents, fitnesses = searcher(step, sample_fn)

        if objective == "max":
            best_idx = fitnesses.index(max(fitnesses))
        elif objective == "min":
            best_idx = fitnesses.index(min(fitnesses))
        best_latent = latents[best_idx]
        best_fitness = fitnesses[best_idx]
        img = sample_fn(best_latent.unsqueeze(0))[0]

        running_time = time.time() - start_time
        if benchmark_cfg.wandb.active:
            wandb_log(
                step,
                img,
                best_fitness,
                np.mean(fitnesses),
                np.median(fitnesses),
                prompt,
                running_time,
                sample_fn.device,
            )

        if benchmark_cfg.save_images.active:
            if isinstance(prompt, list):
                prompt_name = prompt[0][:10]
            else:
                prompt_name = prompt[:10]
            img.save(
                os.path.join(
                    benchmark_cfg.save_images.save_dir, f"{prompt_name}_{step}.png"
                )
            )

        if benchmark_cfg.type == "till_time":
            if running_time >= benchmark_cfg.till_time:
                break
        elif benchmark_cfg.type == "till_steps":
            if step >= benchmark_cfg.for_steps:
                break
        elif benchmark_cfg.type == "till_reward":
            if best_fitness > benchmark_cfg.till_reward:
                break


@hydra.main(config_path="configs/baselines")
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
    if cfg.benchmark.save_images.active:
        subdir_name = f"{cfg.pipeline.type}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        save_dir = os.path.join(cfg.benchmark.save_images.save_dir, subdir_name)
        print(f"Saving images to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        cfg.benchmark.save_images.save_dir = save_dir

    # load eval dataset
    if cfg.dataset.name != "custom":
        dataset = eval_datasets.create_dataset(cfg.dataset)
    else:
        dataset_prompts = cfg.dataset.prompts
        dataset = [{"prompt": prompt} for prompt in dataset_prompts]

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
    sample_fn.add_noise = False  # disable noise addition

    if cfg.dataset.name != "custom":
        data_iter = dataset.iter(batch_size=1)
    else:
        data_iter = dataset

    objective = cfg.fitness.objective # maximize or minimize
    for x in tqdm(data_iter):
        sample_fn.regenerate_latents()
        sample_fn.rembed_text(x["prompt"])
        print(x["prompt"])
        fitness_fn = create_fitness_fn(cfg, x["prompt"])

        # baseline
        baseline_img = sample_fn()
        baseline_fitness = fitness_fn(baseline_img[0])
        img = baseline_img[0]
        if cfg.benchmark.wandb.active:
            wandb.log(
                {
                    "step": 0,
                    "best_img": wandb.Image(img),
                    "prompt": x["prompt"],
                    "pop_best_eval": baseline_fitness.item(),
                    "mean_eval": baseline_fitness.item(),
                    "median_eval": baseline_fitness.item(),
                }
            )

        if cfg.benchmark.save_images.active:
            if isinstance(x["prompt"], list):
                prompt_name = x["prompt"][0][:10]
            else:
                prompt_name = x["prompt"][:10]
            img.save(
                os.path.join(
                    cfg.benchmark.save_images.save_dir,
                    f"{prompt_name}_baseline.png",
                )
            )
        print(f"Baseline fitness: {baseline_fitness}")

        if cfg.solver.algorithm == "random":
            searcher = partial(
                random_search,
                fitness_fn=fitness_fn,
                population=cfg.solver.population,
                split_size=cfg.solver.split_size,
                objective=objective,
            )
        elif cfg.solver.algorithm == "zero_order":
            searcher = ZeroOrderSearch(
                fitness_fn=fitness_fn,
                population=cfg.solver.population,
                split_size=cfg.solver.split_size,
                threshold=cfg.solver.threshold,
                latents=sample_fn.latents,
                objective=objective,
            )

        # run
        benchmark(cfg.benchmark, sample_fn, searcher, x["prompt"], objective)


if __name__ == "__main__":
    main()
