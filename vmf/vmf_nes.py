import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import optax
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from evosax.algorithms.distribution_based.base import Params as BaseParams
from evosax.algorithms.distribution_based.base import State as BaseState
from evosax.algorithms.distribution_based.base import metrics_fn as base_metrics_fn
from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn
from evosax.types import Fitness, Metrics, Population, Solution
from flax import struct
from scipy.stats import vonmises_fisher

from .wood_ulrich import sample_vmf_wood_v2 as sample_vmf_wood


def bessel_ratio(nu, maxterms=1000):
    def _ratio(x):
        ratio = mp.besseli(nu + 1, x, maxterms=maxterms) / mp.besseli(nu, x, maxterms=maxterms)
        return float(ratio)

    return _ratio


#  We are not using this function in the code, sample_vmf_wood is much faster.
def sample_vonmises_fisher(mu, kappa, size, key=None):
    """Sample from the von Mises-Fisher distribution."""
    key = np.random.RandomState(key) if key.any() else None
    return vonmises_fisher.rvs(mu=mu, kappa=kappa.item(), size=size, random_state=key)


@struct.dataclass
class State(BaseState):
    mean: jax.Array
    kappa: jax.Array
    mean_opt_state: optax.OptState
    kappa_opt_state: optax.OptState
    best_solution_kappa: Solution


@struct.dataclass
class Params(BaseParams):
    kappa_init: float


def metrics_fn(
    key: jax.Array,
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
) -> Metrics:
    """Compute metrics for vMF-NES."""
    return base_metrics_fn(key, population, fitness, state, params)


class vMF_NES(DistributionBasedAlgorithm):
    def __init__(
        self,
        population_size: int,
        solution: Solution,
        mean_optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
        kappa_optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
        fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        metrics_fn=metrics_fn,
    ):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        # Optimizer for mean
        self.mean_optimizer = mean_optimizer
        # Optimizer for kappa
        self.kappa_optimizer = kappa_optimizer

        p = self.num_dims / 2 - 1
        self.bessel_ratio = np.vectorize(bessel_ratio(p))

    @property
    def _default_params(self) -> Params:
        return Params(kappa_init=1.0)

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            kappa=jnp.full((1,), params.kappa_init),
            mean_opt_state=self.mean_optimizer.init(jnp.zeros((self.num_dims,))),
            kappa_opt_state=self.kappa_optimizer.init(jnp.zeros((1,))),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_solution_kappa=jnp.full((1,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )
        return state

    def _ask(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[Population, State]:
        # We reparameterize the kappa parameter to avoid numerical issues, kappa' is now exp(kappa).
        kappa_prime = jnp.exp(state.kappa).squeeze()
        print(f"Reparameterized kappa: {kappa_prime}")
        print(f"Original kappa: {state.kappa}")
        print(f"Mean shape: {state.mean.shape}, Kappa shape: {kappa_prime.shape}")
        print(f"Population size: {self.population_size}, num_dims: {self.num_dims}")
        solutions = sample_vmf_wood(key, state.mean, kappa_prime, self.population_size)
        return solutions, state

        # out_type = jax.ShapeDtypeStruct((self.population_size, self.num_dims), state.mean.dtype)
        # solutions = jax.pure_callback(sample_vonmises_fisher, out_type, state.mean, state.kappa, self.population_size, key)
        # return vonmises_fisher.rvs(
        # mu=np.array(state.mean),
        # kappa=np.array(state.kappa),
        # size=self.population_size,
        # random_state=np.random.RandomState(key),
        # )

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        mean = state.mean
        kappa = state.kappa
        kappa = jnp.exp(kappa) # This is due to the reparameterization of kappa

        # Bessel Ratio 
        out_type = jax.ShapeDtypeStruct((1,), jnp.float32)
        # TODO: Figure out how to fix convergence issues with the bessel_ratio function
        # bessel_ratio = jax.pure_callback(self.bessel_ratio, out_type, kappa)
        bessel_ratio = 1

        # Compute gradient of the mean
        mean_score = kappa * (population - (population @ mean)[:, jnp.newaxis] * mean)
        mean_grad = mean_score * fitness[:, jnp.newaxis]
        mean_grad = jnp.mean(mean_grad, axis=0)

        # Compute gradient of the kappa
        kappa_grad = (population @ mean - bessel_ratio)
        kappa_grad = kappa_grad * fitness * kappa # multiply by kappa to account for the reparameterization
        kappa_grad = jnp.mean(kappa_grad, axis=0)

        # FIM mean block
        F_mu_mu = kappa * bessel_ratio * (jnp.eye(self.num_dims) - jnp.outer(mean, mean))

        # FIM kappa block
        F_kappa_kappa = 1 + ((1 - self.num_dims) / kappa) * bessel_ratio - (bessel_ratio**2) / (kappa**2)
        F_kappa_kappa = (kappa**2) * F_kappa_kappa # multiply by kappa^2 to account for the reparameterization

        # Compute natural gradient.
        # TODO: Pinv is very expensive, we should look into more efficient algorithms.
        grad_mean = jnp.linalg.pinv(F_mu_mu, hermitian=True, rcond=0.1) @ mean_grad
        grad_kappa = (F_kappa_kappa ** -1) * kappa_grad

        # Update mean and kappa
        updates_mean, mean_opt_state = self.mean_optimizer.update(grad_mean, state.mean_opt_state)
        mean = optax.apply_updates(state.mean, updates_mean)

        updates_kappa, kappa_opt_state = self.kappa_optimizer.update(grad_kappa, state.kappa_opt_state)
        kappa = optax.apply_updates(state.kappa, updates_kappa)

        return state.replace(
            mean=mean,
            kappa=kappa,
            mean_opt_state=mean_opt_state,
            kappa_opt_state=kappa_opt_state,
        )

