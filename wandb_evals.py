import argparse
import json
import os
from glob import glob

import hpsv2
import ImageReward
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from diffusers.utils import numpy_to_pil, pt_to_pil
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model = (
    CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device=DEVICE)
)


@torch.no_grad()
def clip_score(processor, clip_model, prompt, img):
    inputs = processor(
        text=[prompt], images=[img], return_tensors="pt", padding=True
    ).to(device=DEVICE)
    outputs = clip_model(**inputs)
    score = outputs[0][0]
    return score.tolist()


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
aesthetic_processor = CLIPProcessor(
    CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14"),
    AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
)

aesthetic_clip_model = (
    CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device=DEVICE)
)

### Load linear classifier
aesthetic_mlp = AestheticMLP(768)
aesthetic_mlp.load_state_dict(
    torch.load(os.path.join("./", "aesthetic_mlp.pth"), map_location="cpu"),
    strict=False,
    assign=True,
)
aesthetic_mlp.eval().to(device=DEVICE)


def aesthetic_score(img):
    inputs = aesthetic_processor(
        images=img,
        return_tensors="pt",
        padding=True,
        do_rescale=False,
    ).to(device=DEVICE)

    clip_image_embeddings = aesthetic_clip_model.get_image_features(**inputs)
    score = aesthetic_mlp(clip_image_embeddings)

    return score


def flatten_log(log):
    out = {}
    for k, v in log.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = v2
        elif isinstance(v, list):
            if len(v) == 1:
                out[k] = v[0]
        else:
            out[k] = v
    return out


def handle_v(v):
    if isinstance(v, list):
        return handle_v(v[0])
    if isinstance(v, np.float32):
        return v.item()
    if isinstance(v, torch.Tensor):
        # Handle torch tensors
        if v.numel() == 1:
            return v.item()
        else:
            # Return the numpy array if it has more than one element
            return v.cpu().numpy()
    return v


def parse_args():
    parser = argparse.ArgumentParser(description="WandB Evaluation Script")
    parser.add_argument(
        "--run-path",
        type=str,
        required=True,
        help="WandB run path (e.g., 'user/project/run_id')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save the evaluation results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_path = args.run_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize WandB
    api = wandb.Api(timeout=60)

    # Fetch the run
    run = api.run(run_path)
    run_id = run.id
    scan_history = run.scan_history()  # Fetch the scan history

    # Flatten the logs
    flattened_logs = [flatten_log(log) for log in scan_history]
    all_data = pd.DataFrame(flattened_logs)

    unq_prompts = {i: prompt for i, prompt in enumerate(all_data["prompt"].unique())}
    reverse_unq_prompts = {v: k for k, v in unq_prompts.items()}

    # Save path.
    BASE_SAVE_PATH = output_dir
    # BASE_SAVE_PATH = os.path.join(output_dir, f"{run_id}")
    prompts_json_path = os.path.join(BASE_SAVE_PATH, "prompts.json")
    best_worst_results_path = os.path.join(BASE_SAVE_PATH, "best_worst_results.csv")

    with open(prompts_json_path, "w") as f:
        json.dump(unq_prompts, f)

    # For each prompt, extract the row corresponding to the 0-th step and the step with the maximum pop_best_eval
    best_worst = []
    for prompt, df_group in all_data.groupby("prompt"):
        step_0_row = df_group[df_group["step"] == 0]
        max_pop_best_eval_row = df_group.loc[df_group["pop_best_eval"].idxmax()]
        idx_prompt = reverse_unq_prompts[prompt]

        save_loc = os.path.join(BASE_SAVE_PATH, f"{idx_prompt}")
        os.makedirs(save_loc, exist_ok=True)
        for i, row in df_group.iterrows():
            step = row["step"]
            img_save_loc = os.path.join(save_loc, f"{step}.png")
            if os.path.exists(img_save_loc):
                continue
            try:
                status = run.file(row["best_img.path"]).download()
                os.renames(status.name, os.path.join(save_loc, f"{step}.png"))
            except:
                print(f"Failed to download {row['best_img.path']}")

        baseline_save_loc = os.path.join(save_loc, "baseline.png")
        max_save_loc = os.path.join(save_loc, "max.png")

        if not os.path.exists(baseline_save_loc):
            baseline_status = run.file(step_0_row["best_img.path"].item()).download()
            os.renames(baseline_status.name, os.path.join(save_loc, "baseline.png"))

        if not os.path.exists(max_save_loc):
            max_status = run.file(max_pop_best_eval_row["best_img.path"]).download()
            os.renames(max_status.name, os.path.join(save_loc, "max.png"))
        best_worst.append(
            {
                "prompt": prompt,
                "baseline": step_0_row["pop_best_eval"].item(),
                "best": max_pop_best_eval_row["pop_best_eval"].item(),
            }
        )
        print(
            f"Prompt: {prompt}, Step 0: {step_0_row['pop_best_eval'].item()}, Max Pop Best Eval: {max_pop_best_eval_row['pop_best_eval'].item()}"
        )

    best_worst_results = pd.DataFrame(best_worst)
    best_worst_results.to_csv(best_worst_results_path, index=False)

    img_reward = ImageReward.load("ImageReward-v1.0")
    img_reward = img_reward.eval()

    prompts_file = os.path.join(BASE_SAVE_PATH, "prompts.json")
    # best_worst_file = os.path.join(BASE_SAVE_PATH, "best_worst_results.csv")

    with open(prompts_file, "r") as f:
        prompt_dict = json.load(f)

    measurements = []
    for idx, prompt in tqdm(prompt_dict.items()):
        img_save_loc = os.path.join(BASE_SAVE_PATH, idx)

        max_path = os.path.join(img_save_loc, "max.png")
        baseline_path = os.path.join(img_save_loc, "baseline.png")

        if os.path.exists(max_path):
            img = Image.open(max_path)

            with torch.no_grad():
                img_reward_score = img_reward.score(prompt, img)
                hpsv_reward_score = hpsv2.score(img, prompt)
                aesthetic_reward_score = aesthetic_score(img)
                clip_reward_score = None
                try:
                    with torch.no_grad():
                        clip_reward_score = clip_score(
                            processor, clip_model, prompt, img
                        )
                except:
                    print(f"Failed to score {prompt} with CLIP")

            measurements.append(
                {
                    "prompt": prompt,
                    "img_reward_score": img_reward_score,
                    "hpsv_reward_score": hpsv_reward_score[0],
                    "aesthetic_reward_score": aesthetic_reward_score.item(),
                    "clip_reward_score": clip_reward_score[0] if clip_reward_score else None,
                    "img_path": max_path,
                }
            )
        if os.path.exists(baseline_path):
            img = Image.open(baseline_path)

            with torch.no_grad():
                img_reward_score = img_reward.score(prompt, img)
                hpsv_reward_score = hpsv2.score(img, prompt)
                aesthetic_reward_score = aesthetic_score(img)
                clip_reward_score = None
                try:
                    with torch.no_grad():
                        clip_reward_score = clip_score(
                            processor, clip_model, prompt, img
                        )
                except:
                    print(f"Failed to score {prompt} with CLIP")

            measurements.append(
                {
                    "prompt": prompt,
                    "img_reward_score": img_reward_score,
                    "hpsv_reward_score": hpsv_reward_score[0],
                    "aesthetic_reward_score": aesthetic_reward_score.item(),
                    "clip_reward_score": clip_reward_score[0] if clip_reward_score else None,
                    "img_path": baseline_path,
                }
            )

        for img_path in glob(os.path.join(img_save_loc, "[0-9]*.png")):

            img = Image.open(img_path)

            with torch.no_grad():
                img_reward_score = img_reward.score(prompt, img)
                hpsv_reward_score = hpsv2.score(img, prompt)
                aesthetic_reward_score = aesthetic_score(img)
                clip_reward_score = None
                try:
                    with torch.no_grad():
                        clip_reward_score = clip_score(
                            processor, clip_model, prompt, img
                        )
                except:
                    print(f"Failed to score {prompt} with CLIP")

            measurements.append(
                {
                    "prompt": prompt,
                    "img_reward_score": img_reward_score,
                    "hpsv_reward_score": hpsv_reward_score[0],
                    "aesthetic_reward_score": aesthetic_reward_score.item(),
                    "clip_reward_score": clip_reward_score[0] if clip_reward_score else None,
                    "img_path": img_path,
                }
            )
    measurements_df = pd.DataFrame(measurements)
    measurements_df.to_csv(
        os.path.join(BASE_SAVE_PATH, "measurements.csv"), index=False
    )
    print("Measurements saved to measurements.csv")


if __name__ == "__main__":
    main()
