import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import optax
from functools import partial

from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn as base_metrics_fn,
)
from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn
from evosax.types import Fitness, Metrics, Population, Solution
from flax import struct

from .wood_ulrich import sample_vmf_wood_v2 as sample_vmf_wood


@struct.dataclass
class State(BaseState):
    key: jax.Array
    mean: jax.Array
    kappa: jax.Array
    mean_opt_state: optax.OptState
    kappa_opt_state: optax.OptState
    best_solution: Solution
    best_solution_kappa: Solution
    best_fitness: float
    generation_counter: jax.Array

@struct.dataclass
class Params(BaseParams):
    kappa_init: float = 1.0

def metrics_fn(
    key: jax.Array,
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
) -> Metrics:
    return base_metrics_fn(key, population, fitness, state, params)

class vMF_NES(DistributionBasedAlgorithm):
    """
    Natural-evolution strategy on the von Mises-Fisher manifold
    (unit-sphere search distribution).
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        mean_optimizer: optax.GradientTransformation = optax.sgd(1e-3),
        kappa_optimizer: optax.GradientTransformation = optax.sgd(1e-3),
        fitness_shaping_fn=centered_rank_fitness_shaping_fn,
        metrics_fn=metrics_fn,
    ):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)
        self.mean_opt = mean_optimizer
        self.kappa_opt = kappa_optimizer

    # ------------- initialisation -----------------------------------
    @property
    def _default_params(self) -> Params:
        return Params()

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array, mean_init: Solution, params: Params) -> State:
        mean_init = self._ravel_solution(mean_init)
        mean_init = mean_init / jnp.linalg.norm(mean_init)     # ensure unit norm

        state = State(
            key=key,
            mean=mean_init,
            kappa=jnp.full((1,), jnp.log(params.kappa_init)),  # store log-κ
            mean_opt_state=self.mean_opt.init(mean_init),
            kappa_opt_state=self.kappa_opt.init(jnp.zeros((1,))),
            best_solution=jnp.full_like(mean_init, jnp.nan),
            best_solution_kappa=jnp.full((1,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=jnp.array(0, jnp.int32),
        )
        return state

    def _ask(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[Population, State]:
        key, subkey = jax.random.split(state.key)
        kappa = jnp.exp(state.kappa).squeeze()                 # positive scale
        pop = sample_vmf_wood(subkey, state.mean, kappa, self.population_size)
        return pop, state.replace(key=key)

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
        kappa = jnp.exp(kappa)

        # Bessel Ratio 
        # out_type = jax.ShapeDtypeStruct((1,), jnp.float32)
        # TODO: Figure out how to fix convergence issues with the bessel_ratio function
        # bessel_ratio = jax.pure_callback(self.bessel_ratio, out_type, kappa)
        bessel_ratio = jnp.array(1.0, dtype=jnp.float32)

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
        updates_mean, mean_opt_state = self.mean_opt.update(grad_mean, state.mean_opt_state)
        mean = optax.apply_updates(state.mean, updates_mean)

        updates_kappa, kappa_opt_state = self.kappa_opt.update(grad_kappa, state.kappa_opt_state)
        kappa = optax.apply_updates(state.kappa, updates_kappa)

        return state.replace(
            mean=mean,
            kappa=kappa,
            mean_opt_state=mean_opt_state,
            kappa_opt_state=kappa_opt_state,
        )