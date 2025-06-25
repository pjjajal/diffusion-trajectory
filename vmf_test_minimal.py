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

def vmf_nes(popsize, kappa, solution_shape):
    vmf_nes = vMF_NES(
        population_size=popsize, 
        solution=jnp.zeros(solution_shape)
    )

    # Initialize state
    key = jax.random.key(0)
    key, mean_init_key, init_key = jax.random.split(key, 3)
    params = vmf_nes.default_params
    params = params.replace(kappa_init=kappa)
    mean_init = jax.random.normal(mean_init_key, (1, *solution_shape))

    state = vmf_nes.init(init_key, mean_init, params)
    key_ask, key_tell = jax.random.split(key, 2)

    ### Ask, yield a population
    population, state = vmf_nes.ask(key_ask, state, params)

    ### Tell, update ES distribution
    new_state, metrics = vmf_nes.tell(
        key=key_tell,
        population=population,
        fitness=jnp.ones(popsize),
        state=state,
        params=params
    )

    print("Norm of new_state mean:", jnp.linalg.norm(new_state.mean))
    print("Norm of mean_init:", jnp.linalg.norm(state.mean))
    print(f"New state metrics: {metrics}\n")

    return population, np.array(mean_init)

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
    popsize = 2048
    d = 64
    kappa = 50
    solution_shape = (d,)

    # key = jax.random.key(0)
    # key, init_key = jax.random.split(key, 2)    
    # mean = jax.random.normal(init_key, (1, *solution_shape))

    # x = sample_vmf(popsize, key, kappa, mean)

    # x_norm = jnp.linalg.norm(x, axis=-1)
    # print(f"Sample norms allclose to 1.0: {jnp.allclose(x_norm, 1.0)}")
    
    # mean = mean / jnp.linalg.norm(mean, axis=-1, keepdims=True)
    # y = sample_scipy_vmf(popsize, kappa, np.array(mean).squeeze())

    # print(x.shape)
    # print(y.shape)
    population_vmf_nes, mean_init = vmf_nes(popsize, kappa, solution_shape)
    print(f"Population shape: {population_vmf_nes.shape}")

    mean_init = mean_init / np.linalg.norm(mean_init, axis=-1, keepdims=True)
    population_scipy = sample_scipy_vmf(popsize, kappa, mean_init.squeeze())
    print(f"Population shape (scipy): {population_scipy.shape}")

    # Compute MMD between the two populations
    def compute_mmd(x, y, kernel='rbf', gamma=None):
        x = np.asarray(x)
        y = np.asarray(y)
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        def rbf(a, b):
            sq_dist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
            return np.exp(-gamma * sq_dist)
        k_xx = rbf(x, x)
        k_yy = rbf(y, y)
        k_xy = rbf(x, y)
        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        return mmd

    mmd_pop = compute_mmd(np.array(population_vmf_nes), population_scipy)
    print(f"MMD between populations: {mmd_pop}")
