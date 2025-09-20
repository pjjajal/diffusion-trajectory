# Inference-Time Alignment of Diffusion Models with Evolutionary Algorithms


<!-- Brief Summary -->
# Summary
Diffusion models are state-of-the-art generative models in various domains, yet their samples often fail to satisfy downstream objectives such as safety constraints or domain-specific validity. Existing techniques for alignment require gradients, internal model access, or large computational budgets. We introduce an inference-time alignment framework based on evolutionary algorithms. We treat diffusion models as black-boxes and search their latent space to maximize alignment objectives. Our method enables efficient inference-time alignment for both differentiable and non-differentiable alignment objectives across a range of diffusion models. On the DrawBench and Open Image Preferences benchmark, our EA methods outperform state-of-the-art gradient-based and gradient-free inference-time methods. In terms of memory consumption, we require 55% to 76% lower GPU memory than gradient-based methods. In terms of running-time, we are 72% to 80% faster than gradient-based methods. We achieve higher alignment scores over 50 optimization steps on Open Image Preferences than gradient-based and gradient-free methods.

<!-- Installation Guide -->
# Installation 
```bash
python -m pip install ./
```

<!-- Usage Guide -->
# Usage
## benchmark_hydra.py
As its name implies, `offline_computation.py` identifies a number of tokens to prune *R* according to the offline computation from our work. Given a device and a pre-trained model, it measures the latency-workload relationship for this device-model pair. Then, we compute *R* using a heuristic based on this relationship. You can also control the granularity (and runtime) of the grid-search with (start,stop,stride) parameters.

> [!TIP]
> It is also possible to separate the grid-search for latency and the grid-search for accuracy estimation. 
> For example, you can estimate accuracy with a high batch size on a more powerful device, then measure latency on the target device for a particular batch size.

Below is an example use of `offline_computation.py` for DeiT-S with batch-size=4:
```bash
> sudo bash scripts/jetson_agxorin_set_clocks.sh
> python offline_computation.py --model deit_small_patch16_224 --batch-size 4 --grid-token-start 196 --grid-token-stop 2 --grid-token-stride 1
Loaded model deit_small_patch16_224
...
Computed R=56 given N=197 input tokens
Done!
```

## benchmark_dno.py
TODO

## benchmark_dno_nograd.py
TODO

