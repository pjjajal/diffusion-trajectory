import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from vmf.vmf_nes import vMF_NES
from vmf.wood_ulrich import sample_vmf_wood_v2 as sample_vmf_wood
import optax
from scipy.stats import vonmises_fisher
from scipy.stats import gaussian_kde

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
    return population


def sample_vmf(popsize: int, key: jax.Array, kappa: float | jax.Array, mean: jax.Array) -> jax.Array:
    x = sample_vmf_wood(
        key=key,
        mu=mean,
        kappa=kappa,
        n_samples=popsize
    )

    return x

def sample_scipy_vmf(popsize: int, kappa: int, mean: np.ndarray) -> np.ndarray:
    x = vonmises_fisher.rvs(
        mu=mean,
        kappa=kappa,
        size=popsize
    )

    return x

if __name__ == "__main__":
    popsize = 128
    d = 64
    kappa = 50
    solution_shape = (d,)

    key = jax.random.key(0)
    key, init_key = jax.random.split(key, 2)    
    mean = jax.random.normal(init_key, (1, *solution_shape))

    x = sample_vmf(popsize, key, kappa, mean)

    x_norm = jnp.linalg.norm(x, axis=-1)
    print(f"Sample norms allclose to 1.0: {jnp.allclose(x_norm, 1.0)}")
    
    # mean = mean / jnp.linalg.norm(mean, axis=-1, keepdims=True)
    # y = sample_scipy_vmf(popsize, kappa, np.array(mean).squeeze())

    # print(x.shape)
    # print(y.shape)

    # population = vmf_nes(popsize, solution_shape)
    # print(f"Population shape: {population.shape}")
    # print(f"Population norms: {jnp.linalg.norm(population, axis=-1)}")