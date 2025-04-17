import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline
from diffusers.utils import pt_to_pil, numpy_to_pil
import argparse
import torch.utils.checkpoint as checkpoint
import os
import shutil
from PIL import Image
import time
from torch import autocast
from torch.amp import GradScaler
from dno.rewards import RFUNCTIONS
from fitness import imagereward_gradient_flow_fitness_fn, hpsv2_fitness_fn, hpsv2_gradient_flow_fitness_fn
import numpy as np
import json
import warnings
import eval_datasets
import wandb
from omegaconf import DictConfig
import random

warnings.filterwarnings("ignore")

### This is used to save the config file to wandb
def flatten_dict(d: DictConfig):
    out = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            for k2, v2 in flatten_dict(v).items():
                out[k + "." + k2] = v2
        else:
            out[k] = v
    return out

# sampling algorithm
class SequentialDDIM:
	def __init__(self, timesteps = 100, scheduler = None, eta = 0.0, cfg_scale = 4.0, device = "cuda", opt_timesteps = 50):
		self.eta = eta 
		self.timesteps = timesteps
		self.num_steps = timesteps
		self.scheduler = scheduler
		self.device = device
		self.cfg_scale = cfg_scale
		self.opt_timesteps = opt_timesteps 

		# compute some coefficients in advance
		scheduler_timesteps = self.scheduler.timesteps.tolist()
		scheduler_prev_timesteps = scheduler_timesteps[1:]
		scheduler_prev_timesteps.append(0)
		self.scheduler_timesteps = scheduler_timesteps[::-1]
		scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
		alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]
		alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]

		now_coeff = torch.tensor(alphas_cumprod)
		next_coeff = torch.tensor(alphas_cumprod_prev)
		now_coeff = torch.clamp(now_coeff, min = 0)
		next_coeff = torch.clamp(next_coeff, min = 0)
		m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
		m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
		self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
		self.nl = self.noise_thr * self.eta
		self.nl[0] = 0.
		m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
		self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
		self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x

	def is_finished(self):
		return self._is_finished

	def get_last_sample(self):
		return self._samples[0]

	def prepare_model_kwargs(self, prompt_embeds = None):

		t_ind = self.num_steps - len(self._samples)
		t = self.scheduler_timesteps[t_ind]
   
		model_kwargs = {
			"sample": torch.stack([self._samples[0], self._samples[0]]),
			"timestep": torch.tensor([t, t], device = self.device),
			"encoder_hidden_states": prompt_embeds
		}

		model_kwargs["sample"] = self.scheduler.scale_model_input(model_kwargs["sample"],t)

		return model_kwargs


	def step(self, model_output):
		model_output_uncond, model_output_text = model_output[0].chunk(2)
		direction = model_output_uncond + self.cfg_scale * (model_output_text - model_output_uncond)
		direction = direction[0]

		t = self.num_steps - len(self._samples)

		if t <= self.opt_timesteps:
			now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]
		else:
			with torch.no_grad():
				now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]

		self._samples.insert(0, now_sample)
		
		if len(self._samples) > self.timesteps:
			self._is_finished = True

	def initialize(self, noise_vectors):
		self._is_finished = False

		self.noise_vectors = noise_vectors

		if self.num_steps == self.opt_timesteps:
			self._samples = [self.noise_vectors[-1]]
		else:
			self._samples = [self.noise_vectors[-1].detach()]

def sequential_sampling(pipeline, unet, sampler, prompt_embeds, noise_vectors): 


	sampler.initialize(noise_vectors)

	model_time = 0
	step = 0
	while not sampler.is_finished():
		step += 1
		model_kwargs = sampler.prepare_model_kwargs(prompt_embeds = prompt_embeds)
		#model_output = pipeline.unet(**model_kwargs)
		model_output = checkpoint.checkpoint(unet, model_kwargs["sample"], model_kwargs["timestep"], model_kwargs["encoder_hidden_states"],  use_reentrant=False)
		sampler.step(model_output) 
		
		samp = sampler.get_last_sample().detach()
		samp = sample = decode_latent(pipeline.vae, samp)
		img = to_img(samp)

	return sampler.get_last_sample()

def decode_latent(decoder, latent):
	img = decoder.decode(latent.unsqueeze(0) / 0.18215).sample
	return img

def to_img(img):
	img = torch.clamp(127.5 * img.cpu().float() + 128.0, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()

	return img[0]

def compute_probability_regularization(noise_vectors, eta, opt_time, subsample, shuffled_times = 100):
	
	
	# squential subsampling
	if eta > 0:
		noise_vectors_flat = noise_vectors[:(opt_time + 1)].flatten()
	else:
		noise_vectors_flat = noise_vectors[-1].flatten()
		
	dim = noise_vectors_flat.shape[0]

	# use for computing the probability regularization
	subsample_dim = round(4 ** subsample)
	subsample_num = dim // subsample_dim
		
	noise_vectors_seq = noise_vectors_flat.view(subsample_num, subsample_dim)

	seq_mean = noise_vectors_seq.mean(dim = 0)
	noise_vectors_seq = noise_vectors_seq / np.sqrt(subsample_num)
	seq_cov = noise_vectors_seq.T @ noise_vectors_seq
	seq_var = seq_cov.diag()
	
	# compute the probability of the noise
	seq_mean_M = torch.norm(seq_mean)
	seq_cov_M = torch.linalg.matrix_norm(seq_cov - torch.eye(subsample_dim, device = seq_cov.device), ord = 2)
	
	seq_mean_log_prob = - (subsample_num * seq_mean_M ** 2) / 2 / subsample_dim
	seq_mean_log_prob = torch.clamp(seq_mean_log_prob, max = - np.log(2))
	seq_mean_prob = 2 * torch.exp(seq_mean_log_prob)
	seq_cov_diff = torch.clamp(torch.sqrt(1+seq_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
	seq_cov_log_prob = - subsample_num * (seq_cov_diff ** 2) / 2 
	seq_cov_log_prob = torch.clamp(seq_cov_log_prob, max = - np.log(2))
	seq_cov_prob = 2 * torch.exp(seq_cov_log_prob)

	shuffled_mean_prob_list = []
	shuffled_cov_prob_list = [] 
	
	shuffled_mean_log_prob_list = []
	shuffled_cov_log_prob_list = [] 
	
	shuffled_mean_M_list = []
	shuffled_cov_M_list = []

	for _ in range(shuffled_times):
		noise_vectors_flat_shuffled = noise_vectors_flat[torch.randperm(dim)]   
		noise_vectors_shuffled = noise_vectors_flat_shuffled.view(subsample_num, subsample_dim)
		
		shuffled_mean = noise_vectors_shuffled.mean(dim = 0)
		noise_vectors_shuffled = noise_vectors_shuffled / np.sqrt(subsample_num)
		shuffled_cov = noise_vectors_shuffled.T @ noise_vectors_shuffled
		shuffled_var = shuffled_cov.diag()
		
		# compute the probability of the noise
		shuffled_mean_M = torch.norm(shuffled_mean)
		shuffled_cov_M = torch.linalg.matrix_norm(shuffled_cov - torch.eye(subsample_dim, device = shuffled_cov.device), ord = 2)
		

		shuffled_mean_log_prob = - (subsample_num * shuffled_mean_M ** 2) / 2 / subsample_dim
		shuffled_mean_log_prob = torch.clamp(shuffled_mean_log_prob, max = - np.log(2))
		shuffled_mean_prob = 2 * torch.exp(shuffled_mean_log_prob)
		shuffled_cov_diff = torch.clamp(torch.sqrt(1+shuffled_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
		
		shuffled_cov_log_prob = - subsample_num * (shuffled_cov_diff ** 2) / 2
		shuffled_cov_log_prob = torch.clamp(shuffled_cov_log_prob, max = - np.log(2))
		shuffled_cov_prob = 2 * torch.exp(shuffled_cov_log_prob) 
		
		
		shuffled_mean_prob_list.append(shuffled_mean_prob.item())
		shuffled_cov_prob_list.append(shuffled_cov_prob.item())
		
		shuffled_mean_log_prob_list.append(shuffled_mean_log_prob)
		shuffled_cov_log_prob_list.append(shuffled_cov_log_prob)
		
		shuffled_mean_M_list.append(shuffled_mean_M.item())
		shuffled_cov_M_list.append(shuffled_cov_M.item())
		
	reg_loss = - (seq_mean_log_prob + seq_cov_log_prob + (sum(shuffled_mean_log_prob_list) + sum(shuffled_cov_log_prob_list)) / shuffled_times)
	
	return reg_loss


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Diffusion Optimization with Differentiable Objective')
	parser.add_argument('--model', type=str, default="stable-diffusion-v1-5", help='model name or path')
	parser.add_argument('--fitness-fn', type=str, default="imagereward", choices=["imagereward", "hpsv2"], help='fitness function name')
	parser.add_argument('--num-steps', type=int, default=50, help='number of steps for optimization')
	parser.add_argument('--eta', type=float, default=0.0, help='eta for the DDIM algorithm, eta=0 is ODE-based sampling while eta>0 is SDE-based sampling')
	parser.add_argument('--guidance-scale', type=float, default=7.5, help='guidance scale for classifier-free guidance')
	parser.add_argument('--device', type=str, default="cuda", help='device for optimization')
	parser.add_argument('--seed', type=int, default=42, help='random seed')
	parser.add_argument('--opt-steps', type=int, default=30, help='number of optimization steps')
	parser.add_argument('--opt-time', type=int, default=50, help='number of timesteps in the generation to be optimized')
	parser.add_argument('--precision', choices = ["fp16", "fp32"], default="fp16", help='precision for optimization')
	parser.add_argument('--gamma', type=float, default=0., help='coefficient for the probability regularization')
	parser.add_argument('--subsample', type=int, default=1, help='subsample factor for the computing the probability regularization')
	parser.add_argument('--lr', type=float, default=0.01, help='stepsize for optimization')
	parser.add_argument('--output', type=str, default="output", help='output path')
	### MODIFIED
	parser.add_argument('--cache-dir', type=str, default=None)

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	random.seed(args.seed)

	# load model
	model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
	pipeline = DiffusionPipeline.from_pretrained(
		model_id,
		use_safetensors=True,
		cache_dir=args.cache_dir,
	).to(args.device)

	pipeline.vae.requires_grad_(False)
	pipeline.text_encoder.requires_grad_(False)
	pipeline.unet.requires_grad_(False)
	pipeline.safety_checker = None
	# pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
	pipeline.scheduler.set_timesteps(args.num_steps)
	unet = pipeline.unet

	### Load eval dataset
	dataset_config = DictConfig({"name": "drawbench", "cache_dir": args.cache_dir})
	dataset = eval_datasets.create_dataset(dataset_config)
	dataset_iterator = dataset.iter(batch_size=1)

	### Wandb
	wandb.init(
		project="inference-dno",
		config=vars(args),
	)

	### Output handling
	path_name = f"SD-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
	output_path = os.path.join(args.output, path_name)

	if os.path.exists(output_path):
		shutil.rmtree(output_path)
	os.makedirs(output_path)
	
	### save args
	with open(os.path.join(output_path, "args.json"), "w") as f:
		json.dump(args.__dict__, f, indent = 4)

	# start optimization, opt fpr using fp16 mixed precision
	use_amp = False if args.precision == "fp32" else True
	grad_scaler = GradScaler("cuda", enabled=use_amp, init_scale = 8192)
	amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
	
	for data in dataset_iterator:
		prompt = data["prompt"]
		

		noise_vectors = torch.randn(
			(args.num_steps + 1, 4, 64, 64), 
			generator=torch.Generator(args.device).manual_seed(args.seed), 
			device=args.device
		)
		noise_vectors.requires_grad_(True)
		optimize_groups = [{"params":noise_vectors, "lr":args.lr}]
		optimizer = torch.optim.AdamW(optimize_groups)

		prompt_embeds = pipeline._encode_prompt(
			prompt,
			args.device,
			1,
			True,
		)

		if args.fitness_fn == "hpsv2":
			fitness_callable = hpsv2_gradient_flow_fitness_fn(
				prompt=prompt,
				cache_dir=args.cache_dir, 
				device=args.device
			)
			### Take care with negative sign
			### -1.0x --> Loss
			###  1.0x --> Score
			loss_function = lambda x: torch.mean( -1.0 * fitness_callable(x) )

		elif args.fitness_fn == "imagereward":
			fitness_callable = imagereward_gradient_flow_fitness_fn(
				prompt=prompt,
				cache_dir=args.cache_dir, 
				device=args.device
			)
			### Take care with negative sign
			### -1.0x --> Loss
			###  1.0x --> Score
			loss_function = lambda x: torch.mean( -1.0 * fitness_callable(x) )
		else:
			raise ValueError(f"Unknown fitness function: {args.fitness_fn}")

		print(f"Optimizing for prompt: {prompt}")

		for t in range(args.opt_steps):
			optimizer.zero_grad()

			with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
				ddim_sampler = SequentialDDIM(
					timesteps = args.num_steps,
					scheduler = pipeline.scheduler, 
					eta = args.eta, 
					cfg_scale = args.guidance_scale, 
					device = args.device,
					opt_timesteps = args.opt_time
				)
				
				sample = sequential_sampling(pipeline, unet, ddim_sampler, prompt_embeds = prompt_embeds, noise_vectors = noise_vectors)
				sample = decode_latent(pipeline.vae, sample)
				
				loss = loss_function(sample)
				reward = -loss.item()
				
				if args.gamma > 0:
					reg_loss = compute_probability_regularization(noise_vectors, args.eta, args.opt_time, args.subsample)
					loss = loss + args.gamma * reg_loss

				grad_scaler.scale(loss).backward()
				grad_scaler.unscale_(optimizer)

				torch.nn.utils.clip_grad_norm_([noise_vectors], 1.0)

				grad_scaler.step(optimizer)
				grad_scaler.update()

				wandb.log({
					"step": t,
					"reward": reward,
					"best_img": wandb.Image(sample),
					"prompt": prompt,
				})

		###
		### Save the final image
		###
		ddim_sampler = SequentialDDIM(
			timesteps = args.num_steps,
			scheduler = pipeline.scheduler, 
			eta = args.eta, 
			cfg_scale = args.guidance_scale, 
			device = args.device,
			opt_timesteps = args.opt_time
		)
		
		sample = sequential_sampling(pipeline, unet, ddim_sampler, prompt_embeds = prompt_embeds, noise_vectors = noise_vectors)
		sample = decode_latent(pipeline.vae, sample)
		
		loss = loss_function(sample)
		reward = -loss.item()

		wandb.log({
			"step": args.opt_steps,
			"reward": reward,
			"best_img": wandb.Image(sample),
			"prompt": prompt,
		})

		
		