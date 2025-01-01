import numpy as np
import torch
from diffusers.utils import pt_to_pil
from PIL import Image
from typing import Callable
from transformers import pipeline, CLIPProcessor, CLIPModel

def compose_fitness_fns(fitness_fns: list[Callable], weights: list[float]):
    fitness = lambda img: sum([w * fn(img) for w, fn in zip(weights, fitness_fns)])
    return fitness


def clip_fitness_fn(clip_model_name, prompt, cache_dir=None) -> Callable:
    processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir)
    clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=cache_dir).to(device=0)
    def fitness_fn(img: torch.Tensor) -> float:
        pil_imgs = pt_to_pil(img)
        _prompt = [prompt] if isinstance(prompt, str) else prompt
        # _prompt = [_prompt[0],]
        inputs = processor(text=_prompt, images=pil_imgs, return_tensors="pt", padding=True).to(device=0)
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


def brightness(img: torch.Tensor) -> float:
    pil_imgs = pt_to_pil(img)
    hsv_imgs = [pil_img.convert("HSV") for pil_img in pil_imgs]
    vs = [np.array(hsv_img.split()[-1]) for hsv_img in hsv_imgs]
    v = torch.tensor(np.mean(np.array(vs)) / 255.0).unsqueeze(0)
    return v