import torch
import numpy as np
from einops import einsum
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
from evotorch.algorithms import CMAES, SNES, CEM
from evotorch.logging import PandasLogger
from diffusion_pt import diffusion_sample
from fitness_fn import *
from noise_injection import rotational_transform
from vectorized_problem import VectorizedProblem
import matplotlib.pyplot as plt
import argparse
from PIL.Image import Image
import subprocess
from typing import List, Callable, Dict, Tuple, Any
import logging
import warnings
from pathlib import Path, PosixPath, WindowsPath
from tqdm.auto import tqdm
import pandas


###
### Loggging and output Handling
###
warnings.filterwarnings("ignore")

### Disables evotorch logging
if "evotorch.core" in logging.Logger.manager.loggerDict:
	logging.getLogger("evotorch.core").disabled = True

bench_logger = logging.getLogger("benchmark.py")


###
### Constants, LUTs + Dicts
###
str_to_torch_dtype_LUT = {
	"fp16": torch.float16,
	"fp32": torch.float32,
	"bf16": torch.bfloat16
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Benchmarking')

	parser.add_argument('--seed', type=int, default=37)

	parser.add_argument('--prompt', type=str, default="A picture of a dog.")
	parser.add_argument('--cache-dir', type=str, default=None)
	parser.add_argument('--denoising-steps', type=int, default=50)
	parser.add_argument('--sample-count', type=int, default=20)
	parser.add_argument('--guidance-scale', type=float, default=7.5)

	parser.add_argument('--solver-splits', type=int, default=2)
	parser.add_argument('--mean-scale', type=float, default=0.02)

	parser.add_argument('--fitness-optim-objective', type=str, choices=["max", "min"], default="max")
	parser.add_argument('--fitness-clip-weight', type=float, default=1.0)
	parser.add_argument('--fitness-clip-prompt', type=str, default="A picture of an orange dog.")
	parser.add_argument('--fitness-brightness', type=float, default=0.0)
	parser.add_argument('--fitness-aesthetic-weight', type=float, default=0.0)
	parser.add_argument('--fitness-pick-weight', type=float, default=0.0)
	parser.add_argument('--fitness-novelty-weight', type=float, default=0.0)
	
	parser.add_argument('--pipeline-dtype', type=str, choices=["fp16", "fp32", "bf16"], default="fp16")
	parser.add_argument('--device', type=str, default="cuda:0")
	parser.add_argument('--batch-size', type=int, default=1)

	args = parser.parse_args()

	### Post-process args
	### Replace string values with corresponding torch types
	args.pipeline_dtype = str_to_torch_dtype_LUT[args.pipeline_dtype]
	args.device = torch.device(args.device)

	return args


### Generate a .GIF from a sequence of images 
def ffmpeg_make_gif_subprocess(frames: List[Image], frame_dump_path: Path, gif_filename: str, fps: int = 2, dump_only: bool = False):
	### Sanitize
	assert(gif_filename.lower().endswith(".gif"))

	gif_dump_path = frame_dump_path.joinpath(gif_filename)

	### Dump frames
	for idx, frame in enumerate(frames):
		frame.save(frame_dump_path / f"frame-{idx}.png")
	if dump_only:
		return

	### Invoke ffmpeg
	ffmpeg_in_str = f"{str(frame_dump_path)}/frame-%d.png"
	### Need for Git Bash on windows. Unbelievable.
	ffmpeg_in_str = ffmpeg_in_str.replace("\\", "/") if isinstance(frame_dump_path, WindowsPath) else ffmpeg_in_str
	ffmpeg_out_str = str(gif_dump_path).replace("\\", "/") if isinstance(gif_dump_path, WindowsPath) else gif_dump_path

	### -y allows overwriting
	command = f"ffmpeg -y -framerate {fps} -i {ffmpeg_in_str} {ffmpeg_out_str}"
	bench_logger.info(f"Attempting to execute:\"{command}\"")

	try:
		subprocess.run(
			command, 
			check=True, 
			shell=True,
			stdout=subprocess.DEVNULL,
			stderr=subprocess.STDOUT,
		)
	except subprocess.CalledProcessError as e:
		bench_logger.critical(f"Error invoking ffmpeg:\n{e}\n")


def compose_fitness_callables_and_weights(args: argparse.Namespace) -> Tuple[List[Callable], List[float]]:
	### Fitness Functions
	fitness_callables = []
	fitness_weights = []

	if args.fitness_clip_weight > 0.0:
		fitness_clip_callable = clip_fitness_fn(
			clip_model_name="openai/clip-vit-large-patch14",
			prompt=args.fitness_clip_prompt, 
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
			prompt=["a picture of a dog"], 
			cache_dir=args.cache_dir, 
			device=args.device)
		fitness_callables.append(aesthetic_fitness_callable)
		fitness_weights.append(args.fitness_aesthetic_weight)

	if args.fitness_pick_weight > 0.0:
		pickscore_fitness_callable = pickscore_fitness_fn(
			prompt=["a picture of a dog"], 
			cache_dir=args.cache_dir, 
			device=args.device)
		fitness_callables.append(pickscore_fitness_callable)
		fitness_weights.append(args.fitness_pick_weight)

	if args.fitness_novelty_weight > 0.0:
		novelty_fitness_callable = Novelty(
			prompt=["a picture of a dog"], 
			cache_dir=args.cache_dir, 
			device=args.device)
		fitness_callables.append(novelty_fitness_callable)
		fitness_weights.append(args.fitness_novelty_weight)

	assert(len(fitness_callables) > 0)

	return fitness_callables, fitness_weights


### Iteratively sample and search for a noise vector solution
def diffusion_solve_and_sample(args : argparse.Namespace, solver: CMAES, sample_callable : Callable) -> Tuple[torch.Tensor, pandas.DataFrame]:
	### Get the frames as a single tensor of empty elements
	sampled_frames = torch.empty(
		(1 + args.sample_count, 3, 512, 512), 
		dtype=args.pipeline_dtype, 
		device=args.device
	)

	### Latency Report
	latency_report_dict = {}
	latency_report_columns = ["Sample Index", "Sample Time (ms)", "Solve Time (ms)"]

	### Get the initial image
	sampled_frames[0] = sample_callable()

	### TQDM?
	progress_bar = tqdm(range(args.sample_count), desc="diffusion_solve_and_sample(...)")

	### Solve for updated noise, and resample
	for step in progress_bar:
		solver.step()
		best_candidate_index = solver.population.argbest()
		x = solver.population[best_candidate_index].values
		x = x.reshape(-1, 4 + 4 ** 2)

		mean = x[:, :4]
		cov = x[:, 4:].reshape(-1, 4, 4)
		rot, scaling = torch.linalg.qr(cov)
		mean =  mean.to(args.device, dtype=args.pipeline_dtype).unsqueeze(-1).unsqueeze(-1)
		rot = rot.to(args.device, dtype=args.pipeline_dtype)

		x = einsum(centroid, rot, 'b c h w, p c1 c -> p c1 h w')
		x = x + mean_scale * mean - centroid

		sampled_frames[1 + step] = sample_callable(x)

	### Dict to DataFrame
	latency_report_dataframe = pandas.DataFrame(data=latency_report_dict, columns=latency_report_columns)

	### Return values
	return sampled_frames, latency_report_dataframe


if __name__ == "__main__":
	### Parse arguments
	args = parse_args()
	
	### Set seed
	torch_rng = torch.Generator(args.device).manual_seed(args.seed)
	np.random.seed(args.seed)

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

	sample_callable, latents, _, dtype = diffusion_sample(
		pipeline, 
		prompt=[args.prompt], 
		num_inference_steps=args.denoising_steps, 
		generator=torch_rng, 
		guidance_scale=args.guidance_scale, 
		batch_size=args.batch_size)
	
	###
	### Fitness Setup
	###
	fitness_callables, fitness_weights = compose_fitness_callables_and_weights(args)
	total_fitness_callable = compose_fitness_fns(fitness_callables, fitness_weights)

	###
	### Problem and Solver Setup
	###
	mean_scale = 0.01
	initial_bounds = (-1,1)

	objective_callable, objective_transform_callable, centroid, solution_length = rotational_transform(
		sample_callable, 
		total_fitness_callable, 
		latents.shape, 
		args.device, 
		center=latents, 
		mean_scale=mean_scale, 
		dtype=dtype)
	
	problem = VectorizedProblem(
		objective_sense=args.fitness_optim_objective,
		objective_func=objective_callable, 
		solution_length=solution_length, 
		initial_bounds=initial_bounds, 
		dtype=np.dtype('float32'),
		splits=args.solver_splits, 
		initialization=None)
	
	solver = CMAES(problem, stdev_init=1, separable=False, csa_squared=True)
	solver_logger = PandasLogger(solver)

	###
	### Benchmarking!
	###
	sampled_frames_tensor, latency_report_dataframe = diffusion_solve_and_sample(args, solver, sample_callable)
	solver_report_dataframe = solver_logger.to_dataframe()

	### Convert `sampled_frames` Tensor to PIL Image list
	sampled_frames_image_list = pt_to_pil(sampled_frames_tensor)

	### Sanitize paths
	frame_dump_path = Path("frames")
	frame_dump_path.mkdir(parents=True, exist_ok=True)

	### Dump frames, GIF
	ffmpeg_make_gif_subprocess(sampled_frames_image_list, frame_dump_path, "animation.gif", fps=2, dump_only=False)

	### Dump solver report, latency
	solver_report_dataframe.to_csv("report_solver.csv", index=False)
	latency_report_dataframe.to_csv("report_latency.csv", index=False)

	bench_logger.info("Done!")
