import torch
import torchvision
from torchvision.transforms.functional import center_crop, normalize, resize
from transformers import CLIPImageProcessor
from transformers.utils import TensorType
from transformers.image_processing_base import BatchFeature


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
			raise ValueError(f"Input must be a torch.Tensor: is actually {type(images)}")

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
	

###
### Wrap existing processor to allow gradients
###
def as_tensor_gradient_flow_clip_image_processor(
	processor: CLIPImageProcessor,
) -> CLIPImageProcessorWithTensorGradientFlow:
	processor.__class__ = CLIPImageProcessorWithTensorGradientFlow
	return processor