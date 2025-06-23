import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from vmf.vmf_nes import vMF_NES
import optax

popsize = 4
es = vMF_NES(
    population_size=popsize, 
    solution=jnp.zeros((4, 4 ,4))
)

# Initialize state
key = jax.random.key(0)

key, subkey, init_key = jax.random.split(key, 3)
params = es.default_params
params = params.replace(kappa_init=50.)

mean = jax.random.normal(init_key, (4, 4, 4))
state = es.init(subkey, mean, params)

old_mean, old_kappa = state.mean, state.kappa
key, key_ask, key_norm, key_tell = jax.random.split(key, 4)

population, state = es.ask(key_ask, state, params)

print(f"Population: {population}")

popsize = population.shape[0]
norms = jnp.sqrt(
    jax.random.chisquare(key_norm, es.num_dims, (popsize,))
)
population = population * norms[:, None, None, None]

print(f"Normed  population: {population}")

# torch_population = torch.from_numpy(np.array(population))
