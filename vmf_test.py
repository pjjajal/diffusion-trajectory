import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
from diffusers import StableDiffusionPipeline

from fitness import brightness
from noise_injection_pipelines import SDSamplingPipeline
from vmf.vmf_nes import vMF_NES
import optax

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    device_map=None,
    torch_dtype=torch.float32,
    use_safetensors=True,
    safety_checker=None,
).to("cuda")

sample_fn = SDSamplingPipeline(
    pipeline=pipeline,
    prompt="",
    num_inference_steps=50,
    classifier_free_guidance=True,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(0),
    add_noise=False,
    height=512,
    width=512,
)

sample_fn.regenerate_latents()
sample_fn.rembed_text("An orange cat sitting on a couch")

popsize = 6
es = vMF_NES(
    population_size=popsize, 
    solution=jnp.zeros((4, 64 ,64)),
    mean_optimizer=optax.sgd(10.0),
    kappa_optimizer=optax.sgd(10.0),
)

# Initialize state
key = jax.random.key(0)

key, subkey, init_key = jax.random.split(key, 3)
params = es.default_params
params = params.replace(kappa_init=50.)


mean = jax.random.normal(init_key, (4, 64, 64))
state = es.init(subkey, mean, params)

best_sols = []
for i in range(10):
    old_mean, old_kappa = state.mean, state.kappa
    key, key_ask, key_norm, key_tell = jax.random.split(key, 4)
    print("HERE")
    population, state = es.ask(key_ask, state, params)
    print("THERE")

    popsize = population.shape[0]
    norms = jnp.sqrt(
        jax.random.chisquare(key_norm, es.num_dims, (popsize,))
    )
    population = population * norms[:, None, None, None]
    
    print("HERE")
    torch_population = torch.from_numpy(np.array(population)).to("cuda")
    print("THERE")

    imgs = []
    for latents in torch_population.split(4):
        img = sample_fn(noise_injection=latents)
        imgs.extend(img)

    fitnesses = [brightness(img) for img in imgs]
    idx_max = np.argmax(fitnesses)
    best_img = imgs[idx_max]

    fitnesses = jnp.array(fitnesses).squeeze()
    # Tell the ES about the fitnesses
    state, metrics = es.tell(key_tell, population, fitnesses, state, params)
    # Get the best solution
    best_sol = state.best_solution
    best_sols.append((best_img, state.best_fitness))
    print(f"Iteration {i}")
    print(
        f"Best fitness: {state.best_fitness}",
        f"Mean fitness: {jnp.mean(fitnesses)}",
        f"Max Fitness: {jnp.max(fitnesses)}",
        f"Min Fitness: {jnp.min(fitnesses)}",
        f"KAppa: {state.kappa}",
        f"Delta Mean: {jnp.linalg.norm(old_mean - state.mean)}",
        f"Delta Kappa: {jnp.linalg.norm(old_kappa - state.kappa)}",
    )