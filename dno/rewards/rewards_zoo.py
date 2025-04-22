import torch
import torch.nn as nn
import torchvision
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor, AutoProcessor, AutoModel, AutoTokenizer
from transformers.image_processing_base import BatchFeature
from transformers.utils import TensorType
from torchvision.transforms.functional import center_crop, normalize, resize
import io

###
### Necessary for DNO to have gradients
### STOLEN SHAMELESSY FROM https://github.com/huggingface/transformers/issues/21064
###
class CLIPImageProcessorWithTensorGradientFlow(CLIPImageProcessor):
	"""
	This wraps the huggingface CLIP processor to allow backprop through the image processing step.
	The original processor forces conversion to numpy then PIL images, which is faster for image processing but breaks gradient flow.
	"""
	model_input_names = ["pixel_values"]

	def __init__(
		self,
		**kwargs,
	) -> None:
		super().__init__(**kwargs)

	def preprocess(
		self,
		images,
		do_resize=None,
		size=None,
		resample=None,
		do_center_crop=None,
		crop_size=None,
		do_rescale=None,
		rescale_factor=None,
		do_normalize=None,
		image_mean=None,
		image_std=None,
		**kwargs,
	) -> BatchFeature:
		### Assert!
		if not isinstance(images, torch.Tensor):
			raise ValueError("Input must be a torch.Tensor")

		### Overwrite the default values as necessary
		do_resize = do_resize if do_resize is not None else self.do_resize
		size = size if size is not None else self.size
		resample = resample if resample is not None else self.resample
		do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
		crop_size = crop_size if crop_size is not None else self.crop_size
		do_rescale = do_rescale if do_rescale is not None else self.do_rescale
		rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
		do_normalize = do_normalize if do_normalize is not None else self.do_normalize
		image_mean = image_mean if image_mean is not None else self.image_mean
		image_std = image_std if image_std is not None else self.image_std

		### Process!
		### NOTE: Resize is the primary source of numerical difference
		if do_resize:
			if "shortest_edge" in size:
				size = (size["shortest_edge"],size["shortest_edge"])
			elif "height" in size and "width" in size:
				size = (size["height"], size["width"])
			else:
				raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")
			images = resize(images, size, interpolation=resample, antialias=True)
		if do_center_crop:
			images = center_crop(images, list(self.crop_size.values()))
		if do_rescale:
			images = images * rescale_factor
		if do_normalize:
			images = normalize(images, mean=self.image_mean, std=self.image_std)	

		data = {"pixel_values": images}
		return BatchFeature(data=data, tensor_type=TensorType.PYTORCH)

	def __call__(self, images: torch.Tensor, **kwargs) -> BatchFeature:
		return self.preprocess(images, **kwargs)


# Local Model Path: Change to the model path to your own model path
### MODIFIED
# CLIP_PATH = '/mnt/workspace/workgroup/tangzhiwei.tzw/clip-vit-large-patch14'
CLIP_PATH = "openai/clip-vit-large-patch14"
AESTHETIC_PATH = 'sac+logos+ava1-l14-linearMSE.pth'
HPS_V2_PATH = "/mnt/workspace/workgroup/tangzhiwei.tzw/HPS_v2_compressed.pt"
# PICK_SCORE_PATH = "/mnt/workspace/workgroup/tangzhiwei.tzw/pickscore"
PICK_SCORE_PATH = "yuvalkirstain/PickScore_v1"

# Aesthetic Scorer
class MLPDiff(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(768, 1024),
			nn.Dropout(0.2),
			nn.Linear(1024, 128),
			nn.Dropout(0.2),
			nn.Linear(128, 64),
			nn.Dropout(0.1),
			nn.Linear(64, 16),
			nn.Linear(16, 1),
		)


	def forward(self, embed):
		return self.layers(embed)


class AestheticScorerDiff(torch.nn.Module):
	def __init__(self, dtype):
		super().__init__()
		self.clip = CLIPModel.from_pretrained(CLIP_PATH)
		self.mlp = MLPDiff()
		state_dict = torch.load(AESTHETIC_PATH)
		self.mlp.load_state_dict(state_dict)
		self.dtype = dtype
		self.eval()

	def __call__(self, images):
		device = next(self.parameters()).device
		embed = self.clip.get_image_features(pixel_values=images)
		embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
		return self.mlp(embed).squeeze(1)

def aesthetic_loss_fn(device=None,
					 inference_dtype=None):
	
	target_size = 224
	normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
												std=[0.26862954, 0.26130258, 0.27577711])
	scorer = AestheticScorerDiff(dtype=inference_dtype).to(device)
	scorer.requires_grad_(False)
	target_size = 224
	def loss_fn(im_pix_un, prompts=None):
		im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
		im_pix = torchvision.transforms.Resize(target_size)(im_pix)
		im_pix = normalize(im_pix).to(im_pix_un.dtype)
		rewards = scorer(im_pix)
		loss = -1 * rewards

		return loss
		
	return loss_fn

def white_loss_fn(device=None,
					 inference_dtype=None):
	
	def loss_fn(im_pix_un, prompts=None):
		
		rewards = im_pix_un.mean() 
		loss = -1 * rewards

		return loss
		
	return loss_fn


def black_loss_fn(device=None,
					 inference_dtype=None):
	
	def loss_fn(im_pix_un, prompts=None):
		
		rewards = im_pix_un.mean() 
		loss =  rewards

		return loss
		
	return loss_fn

def contrast_loss_fn(device=None,
					 inference_dtype=None):
	
	def loss_fn(im_pix_un, prompts=None):
		
		rewards = im_pix_un.sum(dim=1).var()
		loss = -1 * rewards

		return loss
		
	return loss_fn

# HPS-v2
def hps_loss_fn(inference_dtype=None, device=None):
	import hpsv2
	from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

	model_name = "ViT-H-14"
	model, preprocess_train, preprocess_val = create_model_and_transforms(
		model_name,
		"/mnt/workspace/workgroup/tangzhiwei.tzw/open_clip_pytorch_model.bin",
		precision=inference_dtype,
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
	
	tokenizer = get_tokenizer(model_name)
	
	checkpoint_path = HPS_V2_PATH
	
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['state_dict'])
	tokenizer = get_tokenizer(model_name)
	model = model.to(device, dtype=inference_dtype)
	model.eval()

	target_size =  224
	normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
												std=[0.26862954, 0.26130258, 0.27577711])
		
	def loss_fn(im_pix_un, prompts):    
		im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
		x_var = torchvision.transforms.Resize(target_size)(im_pix)
		x_var = normalize(x_var).to(im_pix.dtype)        
		caption = tokenizer(prompts)
		caption = caption.to(device)
		outputs = model(x_var, caption)
		image_features, text_features = outputs["image_features"], outputs["text_features"]
		logits = image_features @ text_features.T
		scores = torch.diagonal(logits)
		loss = - scores
		return  loss
	
	return loss_fn

# pickscore
def pick_loss_fn(inference_dtype=None, device=None):
	### Get model and processor from PickScore, custom ImageProcessor
	model = AutoModel.from_pretrained(PICK_SCORE_PATH).eval().to(device=device)
	processor = CLIPProcessor(
		 CLIPImageProcessorWithTensorGradientFlow(
			 size=224,
			 crop_size={"height": 224, "width": 224},
		 ), 
		 AutoTokenizer.from_pretrained(PICK_SCORE_PATH)
	)

	def loss_fn(im_pix_un, prompts):
		image_inputs = processor(
			images=im_pix_un,
			return_tensors="pt",
			padding=True,
		).to(device=device)

		text_inputs = processor(
			text=prompts,
			return_tensors="pt",
			padding=True,
		).to(device=device)

		image_embeddings = model.get_image_features(**image_inputs)
		image_embeddings = image_embeddings / torch.norm(
			image_embeddings, dim=-1, keepdim=True
		)

		text_embeddings = model.get_text_features(**text_inputs)
		text_embeddings = text_embeddings / torch.norm(
			text_embeddings, dim=-1, keepdim=True
		)

		score = model.logit_scale.exp() * (text_embeddings @ image_embeddings.T)[0]
		return  -score
	
	return loss_fn


# CLIP score evaluation
def clip_score(inference_dtype=None, device=None):
	from transformers import CLIPProcessor, CLIPModel

	model = CLIPModel.from_pretrained(CLIP_PATH)
	processor = CLIPProcessor.from_pretrained(CLIP_PATH)
	
	model = model.to(device = device, dtype=inference_dtype)
	
	def loss_fn(image, prompt):    
		inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
		
		for key, value in inputs.items():
			inputs[key] = value.to(device)

		outputs = model(**inputs)
		logits_per_image = outputs.logits_per_image 
		score = logits_per_image.cpu().numpy()[0][0]
		
		return  score
	
	return loss_fn


def jpeg_compressibility(inference_dtype=None, device=None):
	def loss_fn(im_pix_un, prompts = None, **kwargs):
		images = ((im_pix_un / 2) + 0.5).clamp(0, 1)
		if isinstance(images, torch.Tensor):
			images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
			images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
		images = [Image.fromarray(image) for image in images]
		buffers = [io.BytesIO() for _ in images]
		for image, buffer in zip(images, buffers):
			image.save(buffer, format="JPEG", quality=95)
		sizes = [buffer.tell() / 1000 for buffer in buffers]
		return torch.tensor(sizes, dtype=inference_dtype, device=device)

	return loss_fn