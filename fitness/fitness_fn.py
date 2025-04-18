import io
import os
from typing import *

import numpy as np
import torch
import torch.nn as nn
from diffusers.utils import numpy_to_pil, pt_to_pil
from PIL.Image import Image
from transformers import (
	AutoModel,
	AutoProcessor,
	AutoTokenizer,
	CLIPImageProcessor,
	CLIPModel,
	CLIPProcessor,
)

from torchvision.transforms import (
	CenterCrop,
	Compose,
	InterpolationMode,
	Normalize,
	Resize,
	ToTensor,
)

from .hf_gradient_processors import as_tensor_gradient_flow_clip_image_processor
from huggingface_hub import hf_hub_download

try:
	import hpsv2
	from hpsv2.img_score import (
		initialize_model as hpsv2_initialize_model,
		model_dict as hpsv2_model_dict,
	)
	from hpsv2.utils import (
		hps_version_map as hpsv2_huggingface_hub_map,
	)

	from hpsv2.src.open_clip import (
		create_model_and_transforms as hpsv2_create_model_and_transforms,
		get_tokenizer as hpsv2_get_tokenizer,
		OPENAI_DATASET_MEAN,
		OPENAI_DATASET_STD,
	)

except ImportError:
	print(f"HPSv2 not able to be imported, see https://github.com/tgxs002/HPSv2?tab=readme-ov-file#image-comparison for install")
	print(f"Please download the model from https://huggingface.co/tgxs002/HPSv2 and place it in the cache_dir/")

try:
	import ImageReward
except ImportError:
	print(f"Imagereward not able to be imported, see https://github.com/THUDM/ImageReward/tree/main for install")


###
### Type conversion helper
###
def handle_input(img: torch.Tensor | np.ndarray, skip: bool = False) -> Image:
	if skip:
		return img
	if isinstance(img, torch.Tensor):
		pil_imgs = pt_to_pil(img)
	elif isinstance(img, np.ndarray):
		pil_imgs = numpy_to_pil(img)
	else:
		pil_imgs = img
	return pil_imgs


###
### Return callable which computes total fitness score
###
@torch.no_grad()
def compose_fitness_fns(fitness_fns: list[Callable], weights: list[float]) -> Callable:
	assert len(fitness_fns) == len(weights)
	fitness = lambda img: sum(
		[w * fn(img).cpu() for w, fn in zip(weights, fitness_fns)]
	)
	return fitness


###
### CLIP fitness
###
def clip_fitness_fn(
	clip_model_name,
	prompt,
	cache_dir=None,
	device: str = "cpu",
	dtype=torch.float32,
	**kwargs,
) -> Callable:
	processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir)
	clip_model = (
		CLIPModel.from_pretrained(
			clip_model_name, cache_dir=cache_dir, torch_dtype=dtype
		)
		.eval()
		.to(device=device)
	)

	def fitness_fn(img: torch.Tensor | np.ndarray) -> float:
		pil_imgs = handle_input(img)
		_prompt = [prompt] if isinstance(prompt, str) else prompt
		# _prompt = [_prompt[0],]
		inputs = processor(
			text=_prompt, images=pil_imgs, return_tensors="pt", padding=True
		).to(device=device)
		outputs = clip_model(**inputs)
		score = outputs[0][0]
		return score

	return fitness_fn


### Adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py
### Need to manually download https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/6934dd81792f086e613a121dbce43082cb8be85e/sac%2Blogos%2Bava1-l14-linearMSE.pth to cache_dir/
### Adapted for use with Huggingface-hosted CLIP instead of original CLIP package (import clip)
def aesthetic_fitness_fn(
	cache_dir: str = None,
	device: str = "cpu",
	dtype=torch.float32,
	gradient_flow: bool = False,
	**kwargs,
) -> Callable:
	class AestheticMLP(nn.Module):
		def __init__(self, input_size: int):
			super().__init__()
			self.input_size = input_size
			self.layers = nn.Sequential(
				nn.Linear(self.input_size, 1024),
				nn.Dropout(0.2),
				nn.Linear(1024, 128),
				nn.Dropout(0.2),
				nn.Linear(128, 64),
				nn.Dropout(0.1),
				nn.Linear(64, 16),
				nn.Linear(16, 1),
			)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			### Normalize CLIP embedding
			l2 = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
			### NEXT LINE MESSES WITH DNO GRADIENT! There are conflicting implementations (some have this, some dont)
			### Opting to leave it out
			# l2[l2 == 0] = 1
			x = x / l2
			### Apply MLP, return score
			x = self.layers(x)
			return x

	### Load CLIP model and processor
	processor = CLIPProcessor(
		(
			as_tensor_gradient_flow_clip_image_processor(
				CLIPImageProcessor.from_pretrained(
					"openai/clip-vit-large-patch14", cache_dir=cache_dir
				)
			)
			if gradient_flow
			else CLIPImageProcessor.from_pretrained(
				"openai/clip-vit-large-patch14", cache_dir=cache_dir
			)
		),
		AutoTokenizer.from_pretrained(
			"openai/clip-vit-large-patch14", cache_dir=cache_dir
		),
	)

	clip_model = (
		CLIPModel.from_pretrained(
			"openai/clip-vit-large-patch14", cache_dir=cache_dir, torch_dtype=dtype
		)
		.eval()
		.to(device=device)
	)

	### Load linear classifier
	aesthetic_mlp = AestheticMLP(768)
	aesthetic_mlp.load_state_dict(
		torch.load(os.path.join(cache_dir, "sac+logos+ava1-l14-linearMSE.pth")),
		strict=False,
		assign=True,
	)
	aesthetic_mlp.eval().to(device=device)

	def fitness_fn(img: torch.Tensor | np.ndarray) -> float:
		img = handle_input(img, skip=True)
		inputs = processor(
			images=img,
			return_tensors="pt",
			padding=True,
			do_rescale=False,
		).to(device=device)

		clip_image_embeddings = clip_model.get_image_features(**inputs)
		aesthetic_score = aesthetic_mlp(clip_image_embeddings)

		return aesthetic_score

	return fitness_fn


###
### See https://huggingface.co/yuvalkirstain/PickScore_v1
###
def pickscore_fitness_fn(
	prompt: str,
	cache_dir=None,
	device: str = "cpu",
	dtype=torch.float32,
	gradient_flow: bool = False,
	**kwargs,
) -> Callable:
	### Load model and processor
	processor = CLIPProcessor(
		(
			as_tensor_gradient_flow_clip_image_processor(
				CLIPImageProcessor.from_pretrained(
					"yuvalkirstain/PickScore_v1", cache_dir=cache_dir
				)
			)
			if gradient_flow
			else CLIPImageProcessor.from_pretrained(
				"yuvalkirstain/PickScore_v1", cache_dir=cache_dir
			)
		),
		AutoTokenizer.from_pretrained(
			"yuvalkirstain/PickScore_v1", cache_dir=cache_dir
		),
	)

	pick_model = (
		AutoModel.from_pretrained("yuvalkirstain/PickScore_v1", cache_dir=cache_dir)
		.eval()
		.to(device=device)
	)

	def fitness_fn(img: torch.Tensor | np.ndarray) -> float:
		img = handle_input(img, skip=gradient_flow)

		image_inputs = processor(
			images=img,
			return_tensors="pt",
			padding=True,
		).to(device=device)

		text_inputs = processor(
			text=prompt,
			return_tensors="pt",
			padding=True,
		).to(device=device)

		image_embeddings = pick_model.get_image_features(**image_inputs)
		image_embeddings = image_embeddings / torch.norm(
			image_embeddings, dim=-1, keepdim=True
		)

		text_embeddings = pick_model.get_text_features(**text_inputs)
		text_embeddings = text_embeddings / torch.norm(
			text_embeddings, dim=-1, keepdim=True
		)

		score = pick_model.logit_scale.exp() * (text_embeddings @ image_embeddings.T)[0]

		return score

	return fitness_fn

### Taken from ImageReward.py
def imagereward_grad_tensor_transform(size: int = 224):
	### Used pt_to_pil and numpy_to_pil as reference for clamp(...) transform
	return Compose([
		lambda x: ((x / 2) + 0.5).clamp(0, 1),
		Resize(size, interpolation=InterpolationMode.BICUBIC),
		# CenterCrop(size),
		# lambda x: x.convert("RGB"),
		# ToTensor(),
		Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
	])

###
### Imagereward (https://arxiv.org/pdf/2501.09732)
###
def imagereward_fitness_fn(
	prompt: str, cache_dir=None, device: str = "cpu", dtype=torch.float32, **kwargs
) -> Callable:
	
	### Load the model
	imagereward_model = ImageReward.load("ImageReward-v1.0", device=device)
	imagereward_model = imagereward_model.eval()

	def fitness_fn(img: Union[torch.Tensor, Image]) -> float:
		img = handle_input(img)
		rewards = imagereward_model.score(prompt, img)
		return torch.Tensor([rewards])

	return fitness_fn

### Adapted from ImageReward.py to accomodate gradients
def imagereward_gradient_flow_fitness_fn(
	prompt: str, cache_dir=None, device: str = "cpu", dtype=torch.float32, **kwargs
) -> Callable:
	### Load the model
	imagereward_model = ImageReward.load("ImageReward-v1.0", device=device)
	imagereward_model = imagereward_model.eval()
	imagereward_xform = imagereward_grad_tensor_transform(224)
	text_input = imagereward_model.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)

	def fitness_fn(img: torch.Tensor) -> float:
		img_tensor = imagereward_xform(img).to(dtype)
		rewards = imagereward_model.score_gard(
			prompt_attention_mask=text_input.attention_mask, 
			prompt_ids=text_input.input_ids, 
			image=img_tensor
		)
		return rewards

	return fitness_fn

###
### HPSv2 (see https://github.com/tgxs002/HPSv2?tab=readme-ov-file#image-comparison)
###
def hpsv2_grad_tensor_transform(size: int = 224):
	### Used pt_to_pil and numpy_to_pil as reference for clamp(...) transform
	return Compose([
		lambda x: ((x / 2) + 0.5).clamp(0, 1),
		Resize(size, interpolation=InterpolationMode.BICUBIC),
		# CenterCrop(size),
		Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
	])

def hpsv2_fitness_fn(
	prompt: str, cache_dir=None, device: str = "cpu", dtype=torch.float32, **kwargs
) -> Callable:
	prompt = prompt[0] if isinstance(prompt, list) else prompt

	def fitness_fn(img: Union[torch.Tensor, Image]) -> float:
		img = handle_input(img)
		score = hpsv2.score(img, prompt, hps_version="v2.0")

		return torch.Tensor(score)

	return fitness_fn

### HPSv2 with Gradient Flow
def hpsv2_gradient_flow_fitness_fn(
	prompt: str, cache_dir=None, device: str = "cpu", dtype=torch.float32, **kwargs
) -> Callable:
	prompt = prompt[0] if isinstance(prompt, list) else prompt

	### Adapted from HPSV2 internals and original DNO reward_zoo.py
	model_name = "ViT-H-14"
	model, _, _ = hpsv2_create_model_and_transforms(
		model_name,
		'laion2B-s32B-b79K',
		precision=dtype,
		device=device,
		jit=False,
		force_quick_gelu=False,
		force_custom_text=False,
		force_patch_dropout=False,
		force_image_size=None,
		pretrained_image=False,
		image_mean=None,
		image_std=None,
		light_augmentation=True,
		aug_cfg={},
		output_dict=True,
		with_score_predictor=False,
		with_region_predictor=False
	)

	hpsv2_xform = hpsv2_grad_tensor_transform(224)
	checkpoint_path = hf_hub_download("xswu/HPSv2", "HPS_v2_compressed.pt")
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['state_dict'])
	tokenizer = hpsv2_get_tokenizer(model_name)
	model = model.to(device, dtype=dtype)
	model.eval()
		
	def fitness_fn(img: torch.Tensor):    
		img_tensor = hpsv2_xform(img).to(img.dtype)
		caption = tokenizer(prompt).to(device)
		output_dict = model(img_tensor, caption)
		image_features, text_features = output_dict["image_features"], output_dict["text_features"]
		logits = image_features @ text_features.T
		score = torch.diagonal(logits)
		return score
	
	return fitness_fn


def brightness(img: torch.Tensor | np.ndarray, **kwargs) -> float:
	pil_imgs = handle_input(img)
	pil_imgs = [pil_imgs] if not isinstance(pil_imgs, list) else pil_imgs
	hsv_imgs = [pil_img.convert("HSV") for pil_img in pil_imgs]
	vs = [np.array(hsv_img.split()[-1]) for hsv_img in hsv_imgs]
	v = torch.tensor(np.mean(np.array(vs)) / 255.0).unsqueeze(0)
	return v


def relative_luminance(img: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor:
	weights = np.array([0.2126, 0.7152, 0.0722])
	pil_imgs = handle_input(img)
	pil_imgs = [pil_imgs] if not isinstance(pil_imgs, list) else pil_imgs
	imgs = [np.array(pil_img) * weights for pil_img in pil_imgs]
	v = [np.mean(img, axis=(0, 1)).sum() / 255.0 for img in imgs]
	v = torch.tensor(v).unsqueeze(0)
	return v


class Novelty:
	model_dict = {
		"dino_small": ("facebook/dinov2-small", True),
		"dino_base": ("facebook/dinov2-base", True),
		"dino_large": ("facebook/dinov2-large", True),
		"clip_base": ("openai/clip-vit-base-patch16", False),
		"clip_large": ("openai/clip-vit-large-patch14", False),
	}

	def __init__(
		self,
		model_name: Literal[
			"dino_small", "dino_base", "dino_large", "clip_base", "clip_large"
		],
		top_k=5,
		cache_dir=None,
		device=0,
	):
		self.model_name = model_name
		self.cache_dir = cache_dir
		self.device = device
		self.top_k = top_k  # score is computed over the top_k features
		model_url, self.is_dino = self.model_dict[model_name]
		self.processor = AutoProcessor.from_pretrained(model_url, cache_dir=cache_dir)
		self.model = AutoModel.from_pretrained(model_url, cache_dir=cache_dir).to(
			device=device
		)
		self.model = self.model if self.is_dino else self.model.vision_model  # for CLIP
		self.history = None

	def _compute_score(self, x, top_k):
		dists = (x - self.history).norm(dim=-1)
		top_k, _ = torch.topk(dists, k=top_k, largest=False)
		return top_k.mean()

	def __call__(self, img: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor:
		pil_imgs = handle_input(img)
		inputs = self.processor(images=pil_imgs, return_tensors="pt", padding=True).to(
			device=self.device
		)
		outputs = self.model(pixel_values=inputs.pixel_values)
		features = outputs.pooler_output.cpu()
		if self.history is None:
			self.history = features
			return torch.tensor(0.0).unsqueeze(0)
		elif self.history.shape[0] < self.top_k:
			score = self._compute_score(features, self.history.shape[0])
			self.history = torch.cat([self.history, features])
			return score.unsqueeze(0)

		score = self._compute_score(features, self.top_k)
		self.history = torch.cat([self.history, features])
		return score.unsqueeze(0)


def jpeg_compressibility(img: Union[torch.Tensor, Image]) -> torch.Tensor:
	img = handle_input(img)
	buffer = io.BytesIO()
	img.save(buffer, format="JPEG", quality=95)
	size = buffer.tell() / 1000  # size in KB
	buffer.close()
	return torch.Tensor([size])
