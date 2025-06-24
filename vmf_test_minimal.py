import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from vmf.vmf_nes import vMF_NES
from vmf.wood_ulrich import sample_vmf_wood
import optax


def vmf_nes(popsize, solution_shape):
    es = vMF_NES(
        population_size=popsize, 
        solution=jnp.zeros(solution_shape)
    )

    # Initialize state
    key = jax.random.key(0)

    key, subkey, init_key = jax.random.split(key, 3)
    params = es.default_params
    params = params.replace(kappa_init=50.)

    mean = jax.random.normal(init_key, solution_shape)
    state = es.init(subkey, mean, params)

    old_mean, old_kappa = state.mean, state.kappa
    key, key_ask, key_norm, key_tell = jax.random.split(key, 4)

    population, state = es.ask(key_ask, state, params)

    norms = jnp.sqrt(
        jax.random.chisquare(key_norm, es.num_dims, (popsize,))
    )
    population = population * norms[:, None, None, None]

    print(f"Candidate 0 Norm: {jnp.linalg.norm(population[0])}")


def sample_vmf(popsize, solution_shape):
    key = jax.random.key(0)
    key, init_key = jax.random.split(key, 2)    

    mean = jax.random.normal(init_key, (popsize, *solution_shape))
    mean = mean / jnp.linalg.norm(mean, axis=-1, keepdims=True)
    
    x = sample_vmf_wood(
        key=key,
        mu=mean,
        kappa=50.0,
        n_samples=popsize
    )

    return x


# torch_population = torch.from_numpy(np.array(population))
if __name__ == "__main__":
    popsize = 4
    d = 4
    solution_shape = (d,)

    x = sample_vmf(popsize, solution_shape)
    print(f"Sampled vMF: {x}")
    print(f"Sample Norms: {jnp.linalg.norm(x, axis=-1)}")

    # vmf_nes(popsize, solution_shape)
