import numpy as np
import torch
from diffusers.utils import pt_to_pil
from PIL import Image
from typing import Callable, Literal
from transformers import pipeline, CLIPProcessor, CLIPModel, AutoModel, AutoProcessor


def compose_fitness_fns(fitness_fns: list[Callable], weights: list[float]):
    fitness = lambda img: sum([w * fn(img) for w, fn in zip(weights, fitness_fns)])
    return fitness

@torch.no_grad()
def clip_fitness_fn(clip_model_name, prompt, cache_dir=None, device=0) -> Callable:
    processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir)
    clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=cache_dir).to(
        device=device
    )

    def fitness_fn(img: torch.Tensor) -> float:
        pil_imgs = pt_to_pil(img)
        _prompt = [prompt] if isinstance(prompt, str) else prompt
        # _prompt = [_prompt[0],]
        inputs = processor(
            text=_prompt, images=pil_imgs, return_tensors="pt", padding=True
        ).to(device=0)
        outputs = clip_model(**inputs)
        score = outputs[0][0]
        # print(outputs.logits_per_image)
        # score = 0
        # for output in outputs[0]:
        #     if output['label'] == _prompt[0]:
        #         score += output['score']
        #     else:
        #         score -= output['score']
        return score

    return fitness_fn


def brightness(img: torch.Tensor) -> torch.Tensor:
    pil_imgs = pt_to_pil(img)
    hsv_imgs = [pil_img.convert("HSV") for pil_img in pil_imgs]
    vs = [np.array(hsv_img.split()[-1]) for hsv_img in hsv_imgs]
    v = torch.tensor(np.mean(np.array(vs)) / 255.0).unsqueeze(0)
    return v


def relative_luminance(img: torch.Tensor) -> torch.Tensor:
    weights = np.array([0.2126, 0.7152, 0.0722])
    pil_imgs = pt_to_pil(img)
    imgs = [np.array(pil_img) * weights for pil_img in pil_imgs]
    v = [np.mean(img, axis=(0, 1)).sum() / 255.0 for img in imgs]
    v = torch.tensor(v).unsqueeze(0)
    return v


class Novelty:
    model_dict = {
        "dino_small": ("facebook/dinov2-small", True),
        "dino_base": ("facebook/dinov2-base", True),
        "dino_large": ("facebook/dinov2-large", True),
    }

    def __init__(
        self,
        model_name: Literal["dino_small", "dino_base", "dino_large"],
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

    def _compute_score(self, x):
        dists = (x - self.history).norm(dim=-1)
        top_k, _ = torch.topk(dists, k=self.top_k, largest=True)
        return top_k.mean()

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        pil_imgs = pt_to_pil(img)
        inputs = self.processor(images=pil_imgs, return_tensors="pt", padding=True).to(
            device=self.device
        )
        outputs = self.model(pixel_values=inputs.pixel_values)
        features = outputs.pooler_output.cpu()
        if self.history is None:
            self.history = features
            return torch.tensor(0.0).unsqueeze(0)
        elif self.history.shape[0] < self.top_k:
            self.history = torch.cat([self.history, features])
            return torch.tensor(0.0).unsqueeze(0)

        score = self._compute_score(features)
        self.history = torch.cat([self.history, features])
        return score