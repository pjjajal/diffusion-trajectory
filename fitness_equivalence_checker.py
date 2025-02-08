import torch
import numpy as np
from PIL import Image
import warnings
from fitness.fitness_fn import pickscore_fitness_fn, aesthetic_fitness_fn, hpsv2_fitness_fn, imagereward_fitness_fn
from dno.rewards.rewards_zoo import aesthetic_loss_fn, hps_loss_fn, pick_loss_fn, white_loss_fn
import argparse
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel, AutoTokenizer


warnings.filterwarnings("ignore")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Fitness Equivalence Checker")
	parser.add_argument('--image-path', type=str, default="le_goose.png")
	parser.add_argument('--cache-dir', type=str, default="D://Huggingface")
	args = parser.parse_args()

	# processor = AutoProcessor.from_pretrained(
	# 	"yuvalkirstain/PickScore_v1", cache_dir=args.cache_dir
	# )

	# print(processor)

	prompt = "A photo of a dog"
	device = torch.device("cuda:0")
	dtype = torch.float32

	dummy_image = Image.open(args.image_path)
	dummy_image = torch.tensor(np.array(dummy_image), device=device, dtype=dtype, requires_grad=True).permute(2, 0, 1).unsqueeze(0)

	### Input image checker
	print(f"Input image shape: {dummy_image.shape}")
	print(f"Input image max/min: {dummy_image.max()}/{dummy_image.min()}")

	with torch.inference_mode():
		### PickScore
		our_fitness_callable = pickscore_fitness_fn(prompt, cache_dir=args.cache_dir, device=device, dtype=dtype)
		dno_fitness_callable = pick_loss_fn(device=device, inference_dtype=dtype)

		# our_fitness_callable = aesthetic_fitness_fn(cache_dir=args.cache_dir, device=device, dtype=dtype)
		# dno_fitness_callable = aesthetic_loss_fn(device=device, inference_dtype=dtype)

		our_fitness = our_fitness_callable(dummy_image)
		dno_fitness = -dno_fitness_callable(dummy_image, prompt)

		# our_fitness = 0
		# dno_fitness = 0

		print(f"Fitness: {our_fitness} (ours) vs. {dno_fitness} (DNO)")

	exit(0)