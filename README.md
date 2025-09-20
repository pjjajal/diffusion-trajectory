# Inference-Time Alignment of Diffusion Models with Evolutionary Algorithms


<!-- Brief Summary -->
# Summary
Diffusion models are state-of-the-art generative models in various domains, yet their samples often fail to satisfy downstream objectives such as safety constraints or domain-specific validity. Existing techniques for alignment require gradients, internal model access, or large computational budgets. We introduce an inference-time alignment framework based on evolutionary algorithms. We treat diffusion models as black-boxes and search their latent space to maximize alignment objectives. Our method enables efficient inference-time alignment for both differentiable and non-differentiable alignment objectives across a range of diffusion models. On the DrawBench and Open Image Preferences benchmark, our EA methods outperform state-of-the-art gradient-based and gradient-free inference-time methods. In terms of memory consumption, we require 55% to 76% lower GPU memory than gradient-based methods. In terms of running-time, we are 72% to 80% faster than gradient-based methods. We achieve higher alignment scores over 50 optimization steps on Open Image Preferences than gradient-based and gradient-free methods.

<!-- Installation Guide -->
# Installation 
Create a virtual environment (or not) and install dependencies via `requirements.txt`
```bash
python -m pip install -r requirements.txt
```
> [!CAUTION] 
> If using Aesthetic, HPSv2, or ImageReward, additional steps must be taken, which are detailed in `fitness/fitness_fn.py`
> See relevant code below:
```python
...
except ImportError:
    print(
        f"HPSv2 not able to be imported, see https://github.com/tgxs002/HPSv2?tab=readme-ov-file#image-comparison for install"
    )
    print(
        f"Please download the model from https://huggingface.co/tgxs002/HPSv2 and place it in the cache_dir/"
    )

try:
    import ImageReward
except ImportError:
    print(
        f"Imagereward not able to be imported, see https://github.com/THUDM/ImageReward/tree/main for install"
    )
...
```

# Directory Structure & Notes
**Datasets:** The `eval_datasets/open_img_pref` folder contains the subset of 60 Open Image Preferences prompts we used.

**Direct Noise Optimization (DNO):** We re-implemented DNO for evaluation with our method, specifically we modified the code such that our reward function implementations can be used with their approach, for "fair" evaluation and benchmarking.

**Outputs:** Output from `benchmark_hydra.py` will be dumped to `outputs/` under a timestamped directory, specific to the current run.

<!-- Usage Guide -->
# Usage
## benchmark_hydra.py
This is the main evaluation script for our method.
Below is an example use of `benchmark_hydra.py` using the `configs/sd.yaml` configuration.

```bash
> python benchmark_dno.py -cn sd
```

## benchmark_dno.py
> [!CAUTION] 
> TODO

## benchmark_dno_nograd.py
> [!CAUTION] 
> TODO

