import torch
import numpy as np
from einops import einsum
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil, export_to_gif
from evotorch.algorithms import CMAES, SNES, CEM
from noise_injection_pipelines.diffusion_pt import DiffusionSample
from fitness.fitness_fn import *
from noise_injection_pipelines.noise_injection import rotational_transform
from evo.vectorized_problem import VectorizedProblem
import argparse
from PIL.Image import Image
import subprocess
from typing import *
import logging
import warnings
from pathlib import Path, PosixPath, WindowsPath
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import pandas
import time
import wandb

warnings.filterwarnings("ignore")

### Constants, LUTs + Dicts
str_to_torch_dtype_LUT = {
	"fp16": torch.float16,
	"fp32": torch.float32,
	"bf16": torch.bfloat16
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Benchmarking')

	parser.add_argument('--seed', type=int, default=37)

	parser.add_argument('--benchmark-until-fitness', type=float, default=None, help="Benchmark until the total fitness reaches X")
	parser.add_argument('--benchmark-until-time', type=float, default=None, help="Benchmark for X seconds, then report the final total fitness")

	parser.add_argument('--prompt', type=str, default="A picture of a dog.")
	parser.add_argument('--cache-dir', type=str, default=None)
	parser.add_argument('--denoising-steps', type=int, default=50)
	parser.add_argument('--sample-count', type=int, default=20)
	parser.add_argument('--guidance-scale', type=float, default=7.5)

	parser.add_argument('--solver-splits', type=int, default=2)
	parser.add_argument('--mean-scale', type=float, default=0.02)

	parser.add_argument('--fitness-optim-objective', type=str, choices=["max", "min"], default="max")
	parser.add_argument('--fitness-clip-weight', type=float, default=1.0)
	parser.add_argument('--fitness-prompt', type=str, default="A picture of an orange dog.")
	parser.add_argument('--fitness-brightness', type=float, default=0.0)
	parser.add_argument('--fitness-aesthetic-weight', type=float, default=0.0)
	parser.add_argument('--fitness-pick-weight', type=float, default=0.0)
	parser.add_argument('--fitness-novelty-weight', type=float, default=0.0)
	parser.add_argument('--fitness-hpsv2-weight', type=float, default=0.0)
	parser.add_argument('--fitness-imagereward-weight', type=float, default=0.0)
	
	parser.add_argument('--pipeline-dtype', type=str, choices=["fp16", "fp32", "bf16"], default="fp16")
	parser.add_argument('--device', type=str, default="cuda:0")
	parser.add_argument('--batch-size', type=int, default=1)

	parser.add_argument('--gif-dump', action="store_true", default=True)
	parser.add_argument('--gif-dump-fps', type=int, default=2)

	args = parser.parse_args()

	### Validate
	if args.benchmark_until_fitness is None and args.benchmark_until_time is None:
		print(f"WARNING: No benchmark mode specified. Defaulting to optimize over --sample-count steps ({args.sample_count}), then stop.")

	### Post-process args
	### Replace string values with corresponding torch types
	args.pipeline_dtype = str_to_torch_dtype_LUT[args.pipeline_dtype]
	args.device = torch.device(args.device)

	return args


###
### Parse arguments, return callable to return total fitness score
###
def compose_fitness_callables_and_weights(args: argparse.Namespace) -> Callable:
	### Fitness Functions
	fitness_callables = []
	fitness_weights = []

	if args.fitness_clip_weight > 0.0:
		fitness_clip_callable = clip_fitness_fn(
			clip_model_name="openai/clip-vit-large-patch14",
			prompt=args.fitness_prompt, 
			cache_dir=args.cache_dir, 
			device=args.device)
		fitness_callables.append(fitness_clip_callable)
		fitness_weights.append(args.fitness_clip_weight)

	if args.fitness_brightness > 0.0:
		fitness_brightness_callable = brightness()
		fitness_callables.append(fitness_brightness_callable)
		fitness_weights.append(args.fitness_brightness)

	if args.fitness_aesthetic_weight > 0.0:
		aesthetic_fitness_callable = aesthetic_fitness_fn(
			prompt=[args.fitness_prompt], 
			cache_dir=args.cache_dir, 
			device=args.device)
		fitness_callables.append(aesthetic_fitness_callable)
		fitness_weights.append(args.fitness_aesthetic_weight)

	if args.fitness_pick_weight > 0.0:
		pickscore_fitness_callable = pickscore_fitness_fn(
			prompt=[args.fitness_prompt], 
			cache_dir=args.cache_dir, 
			device=args.device)
		fitness_callables.append(pickscore_fitness_callable)
		fitness_weights.append(args.fitness_pick_weight)

	if args.fitness_novelty_weight > 0.0:
		novelty_fitness_callable = Novelty(
			prompt=[args.fitness_prompt], 
			cache_dir=args.cache_dir, 
			device=args.device)
		fitness_callables.append(novelty_fitness_callable)
		fitness_weights.append(args.fitness_novelty_weight)
	
	if args.fitness_hpsv2_weight > 0.0:
		hpsv2_fitness_callable = hpsv2_fitness_fn(
			prompt=args.fitness_prompt, 
			cache_dir=args.cache_dir, 
			device=args.device)
		fitness_callables.append(hpsv2_fitness_callable)
		fitness_weights.append(args.fitness_hpsv2_weight)

	if args.fitness_imagereward_weight > 0.0:
		imagereward_fitness_callable = imagereward_fitness_fn(
			prompt=args.fitness_prompt, 
			cache_dir=args.cache_dir, 
			device=args.device)
		fitness_callables.append(imagereward_fitness_callable)
		fitness_weights.append(args.fitness_imagereward_weight)

	assert(len(fitness_callables) > 0)

	return compose_fitness_fns( fitness_callables, fitness_weights )


def measure_torch_device_memory_used_mb(device: torch.device) -> float:
	if device.type == "cuda":
		free, total = torch.cuda.mem_get_info(device)
		return (total - free) / 1024 ** 2
	else:
		return -1.0


### Iteratively sample and search for a noise vector solution
def diffusion_solve_and_sample(
		args: argparse.Namespace, 
		solver: CMAES, 
		sample_callable: Callable, 
		transform_callable: Callable, 
		total_fitness_callable: Callable
	) -> Tuple:
	### Get the frames as a single tensor of empty elements
	sampled_frames = []

	### Latency Report
	# perf_report_dict = {}
	# perf_report_columns = ["Sample Index", "Elapsed Wall Time(s)", f"Total Fitness", "PyTorch Memory Usage (MB)"]

	### TQDM?
	# progress_bar = tqdm(range(args.sample_count))

	### Step counter, loop condition
	continue_solving = True
	step = 1

	### Operate in inference mode
	with torch.inference_mode():
		### Begin timing
		start_time = time.time()

		### Get the initial image
		sampled_frames.append( sample_callable().cpu() )
		# perf_report_dict[0] = [0, time.time() - start_time, total_fitness_callable(sampled_frames[0]).item(), measure_torch_device_memory_used_mb(args.device)]

		### Solve for updated noise, and resample
		# for step in progress_bar:
		while continue_solving:
			### Solve for updated noise
			solver.step()

			### Identify best candidate noise transformation vector
			best_candidate_index = solver.population.argbest()
			x = solver.population[best_candidate_index].values

			### Store on CPU
			sampled_frames.append( transform_callable(x).squeeze(0).permute(1, 2, 0).detach().cpu().numpy() )

			### Report fitness (slice in a way to ensure the 0th dimension is preserved)
			total_fitness = total_fitness_callable(sampled_frames[step]).item()

			solver_running_wall_time = time.time() - start_time

			wandb.log({
					"Step": step,
					"Total Fitness": total_fitness, 
					"Elapsed Wall Time(s)": solver_running_wall_time, 
					"GPU Memory In-Use(MB)": measure_torch_device_memory_used_mb(args.device),
					"Sampled Image": wandb.Image(sampled_frames[step]),
			})

			### Increment step
			step += 1

			### Evaluate exit condition
			if args.benchmark_until_fitness is not None:
				continue_solving = continue_solving and total_fitness < args.benchmark_until_fitness 
			elif args.benchmark_until_time is not None:
				continue_solving = continue_solving and solver_running_wall_time < args.benchmark_until_time
			else:
				continue_solving = continue_solving and step < args.sample_count
	
	### Dict to DataFrame
	# perf_report_dataframe = pandas.DataFrame.from_dict(data=perf_report_dict, orient="index", columns=perf_report_columns)
	# perf_report_dataframe = perf_report_dataframe.round(2)
	perf_report_dataframe = None

	### Convert List of Tensor to Tensor
	sampled_frames_tensor = torch.stack(sampled_frames).squeeze(1)
	print(sampled_frames_tensor.shape)

	### Return values
	return sampled_frames_tensor, perf_report_dataframe


if __name__ == "__main__":
	### Parse arguments
	args = parse_args()
	
	### Set seed
	torch_rng = torch.Generator(args.device).manual_seed(args.seed)
	np.random.seed(args.seed)

	### Wandb Init
	wandb.init(
		project="inference-diffusion-noise-optim",
		config={
			"seed": args.seed,
			"benchmark_until_fitness": args.benchmark_until_fitness,
			"benchmark_until_time": args.benchmark_until_time,
			"prompt": args.prompt,
			"cache_dir": args.cache_dir,
			"denoising_steps": args.denoising_steps,
			"sample_count": args.sample_count,
			"guidance_scale": args.guidance_scale,
			"solver_splits": args.solver_splits,
			"mean_scale": args.mean_scale,
			"fitness_optim_objective": args.fitness_optim_objective,
			"fitness_clip_weight": args.fitness_clip_weight,
			"fitness_prompt": args.fitness_prompt,
			"fitness_brightness": args.fitness_brightness,
			"fitness_aesthetic_weight": args.fitness_aesthetic_weight,
			"fitness_pick_weight": args.fitness_pick_weight,
			"fitness_novelty_weight": args.fitness_novelty_weight,
			"fitness_hpsv2_weight": args.fitness_hpsv2_weight,
			"fitness_imagereward_weight": args.fitness_imagereward_weight,
			"pipeline_dtype": str(args.pipeline_dtype),
			"device": str(args.device),
			"gpu": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else "N/A",
			"batch_size": args.batch_size,
			"gif_dump_fps": args.gif_dump_fps
		}
	)

	###
	### Diffusion Pipeline Setup
	###
	model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

	pipeline = DiffusionPipeline.from_pretrained(
		model_id,
		use_safetensors=True,
		torch_dtype=args.pipeline_dtype,
		cache_dir=args.cache_dir,
	).to(args.device)

	diffusion_sampler_callable = DiffusionSample(
		pipeline=pipeline,
		prompt=[args.prompt],
		num_inference_steps=args.denoising_steps,
		generator=torch_rng,
		guidance_scale=args.guidance_scale,
		batch_size=args.batch_size
	)

	latents, num_inference_steps, _ = diffusion_sampler_callable.noise_injection_args()
	
	### Fitness Setup
	total_fitness_callable = compose_fitness_callables_and_weights(args)

	### Problem and Solver Setup
	mean_scale = 0.01
	initial_bounds = (-1,1)

	objective_callable, objective_transform_callable, centroid, solution_length = rotational_transform(
		diffusion_sampler_callable, 
		total_fitness_callable, 
		latents.shape, 
		args.device, 
		center=latents, 
		mean_scale=mean_scale, 
		dtype=args.pipeline_dtype)
	
	problem = VectorizedProblem(
		objective_sense=args.fitness_optim_objective,
		objective_func=objective_callable, 
		solution_length=solution_length, 
		initial_bounds=initial_bounds, 
		dtype=np.dtype('float32'),
		splits=args.solver_splits, 
		initialization=None)
	
	solver = CMAES(problem, stdev_init=1, separable=False, csa_squared=True)

	### Benchmarking!
	sampled_frames_tensor, latency_report_dataframe = diffusion_solve_and_sample(args, solver, diffusion_sampler_callable, objective_transform_callable, total_fitness_callable)

	### Cleanup time
	if args.gif_dump:
		sampled_frames_image_list = pt_to_pil(sampled_frames_tensor)
		frame_dump_path = Path("frames")
		frame_dump_path.mkdir(parents=True, exist_ok=True)
		export_to_gif(sampled_frames_image_list, frame_dump_path / "animation.gif", fps=args.gif_dump_fps)

	wandb.finish()

	### Dump solver report, latency
	# report_path = "report.csv"
	# latency_report_dataframe.to_csv(report_path, index=False)
	# print(f"Report dumped to: {report_path}")

	exit(0)
