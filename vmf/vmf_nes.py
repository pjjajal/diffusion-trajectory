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


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
@struct.dataclass
class State(BaseState):
    key: jax.Array
    mean: jax.Array # unit vector (d,)
    kappa: jax.Array # log-κ  (1,)
    mean_opt_state: optax.OptState
    kappa_opt_state: optax.OptState
    best_solution_kappa: Solution


@struct.dataclass
class Params(BaseParams):
    kappa_init: float = 1.0          # positive


# ---------------------------------------------------------------------
# Metric hook
# ---------------------------------------------------------------------
def metrics_fn(
    key: jax.Array,
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
) -> Metrics:
    return base_metrics_fn(key, population, fitness, state, params)


# ---------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------
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

    # ------------- ask ----------------------------------------------
    def _ask(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[Population, State]:
        key, subkey = jax.random.split(state.key)
        kappa = jnp.exp(state.kappa).squeeze()                 # positive scale
        pop = sample_vmf_wood(subkey, state.mean, kappa, self.population_size)
        return pop, state.replace(key=key)

    # ------------- tell ---------------------------------------------
    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # --- un-pack & pre-compute ----------------------------------
        mean   = state.mean
        log_k  = state.kappa
        kappa  = jnp.exp(log_k)                                # scalar

        # Bessel ratio I_{ν+1}/I_ν  (ν = d/2-1)
        nu          = (self.num_dims / 2) - 1
        br          = jsp.ive(nu + 1, kappa) / jsp.ive(nu, kappa)

        # fitness shaping (already centred-ranked by outer loop)
        fit = fitness[:, None]                                 # (N,1)

        # --- score functions ----------------------------------------
        centred = population - (population @ mean)[:, None] * mean
        mean_score  = kappa * centred                          # (N,d)
        kappa_score = (population @ mean)[:, None] - br        # (N,1)

        # --- gradients (sample mean) -------------------------------
        g_mu = jnp.mean(mean_score  * fit, axis=0)             # (d,)
        g_k  = jnp.mean(kappa_score * fit * kappa, axis=0)     # (1,)  (log-κ)

        # --- natural-gradient pre-conditioning ----------------------
        α = kappa * br                                         # scalar
        I   = jnp.eye(self.num_dims, mean.dtype)
        #  F_μμ = α(I − μμᵀ)  →  inverse via Sherman–Morrison
        F_inv_mu = (1/α) * (I + (α / (1 - α)) * jnp.outer(mean, mean))
        ngrad_mu = F_inv_mu @ g_mu
        #  F_κκ for log-κ is simply 1 / Var[κ] ≈ 1  (exact form below)
        F_kappa = 1 + ((1 - self.num_dims) / kappa) * br - (br**2) / (kappa**2)
        ngrad_k = g_k / (kappa**2 * F_kappa)

        # --- optimiser updates --------------------------------------
        upd_mu, mu_state = self.mean_opt.update(ngrad_mu, state.mean_opt_state)
        new_mu = optax.apply_updates(mean, upd_mu)
        new_mu = new_mu / jnp.linalg.norm(new_mu)              # keep on sphere

        upd_k, k_state = self.kappa_opt.update(ngrad_k, state.kappa_opt_state)
        new_log_k = optax.apply_updates(log_k, upd_k)

        return state.replace(
            mean=new_mu,
            kappa=new_log_k,
            mean_opt_state=mu_state,
            kappa_opt_state=k_state,
            generation_counter=state.generation_counter + 1,
        )
