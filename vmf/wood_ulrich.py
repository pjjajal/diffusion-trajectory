import jax
import jax.numpy as jnp
from jax import lax, random
from functools import partial

# ---------------------------------------------------------------------
# helper – Beta(a,b) via the Gamma trick (JAX <1.6 has no random.beta)
# ---------------------------------------------------------------------
def _beta(key, a, b, *, shape):
    k1, k2 = random.split(key)
    ga = random.gamma(k1, a, shape=shape)
    gb = random.gamma(k2, b, shape=shape)
    return ga / (ga + gb)

# ---------------------------------------------------------------------
# Wood-Ulrich vMF sampler (vectorised + fully JIT-able)
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnames=("n_samples",))
def sample_vmf_wood_v2(
    key: jax.Array,
    mu: jax.Array,        # (..., d)
    kappa: jax.Array,     # (...,)
    n_samples: int,
) -> jax.Array:
    """
    Draw `n_samples` iid samples per batch element from vMF(mu, kappa).

    Shapes
    ------
    mu        (..., d)
    kappa     (...,)
    returns   (n_samples, ..., d)
    """
    mu_norm = jnp.linalg.norm(mu, axis=-1, keepdims=True)
    mu = jnp.where(mu_norm > 0, mu / mu_norm, jax.nn.one_hot(0, mu.shape[-1]))

    *batch, d = mu.shape
    batch_shape = tuple(batch)
    kappa = jnp.broadcast_to(kappa, batch_shape)

    # Wood constants
    b  = (-2.0 * kappa + jnp.sqrt(4.0 * kappa**2 + (d - 1.0)**2)) / (d - 1.0)
    x0 = (1.0 - b) / (1.0 + b)
    c  = kappa * x0 + (d - 1.0) * jnp.log1p(-x0**2)

    a_beta = (d - 1.0) / 2.0 # Beta(α, α)

    def body(state):
        key, w, mask, counter = state
        key, k_beta, k_u = random.split(key, 3)

        z = _beta(k_beta, a_beta, a_beta,shape=(n_samples, *batch_shape))
        w_prop = (1.0 - (1.0 + b)[None, ...] * z) / (1.0 - (1.0 - b)[None, ...] * z)
        u = random.uniform(k_u, shape=(n_samples, *batch_shape))

        accept = (
            kappa[None, ...] * w_prop
            + (d - 1.0) * jnp.log1p(-x0[None, ...] * w_prop)
            - c[None, ...]
            >= jnp.log(u)
        )

        w = jnp.where(accept & mask, w_prop, w)
        mask = jnp.where(accept, False, mask)
        return key, w, mask, counter + 1

    def cond(state):
        _, _, mask, counter = state
        return jnp.logical_and(jnp.any(mask), counter < 2048)

    key, k_loop = random.split(key)
    init_state = (
        k_loop,
        jnp.zeros((n_samples, *batch_shape)),
        jnp.ones ((n_samples, *batch_shape), dtype=bool),
        jnp.array(0, jnp.int32),           # rejection counter
    )

    _, w_final, _, _ = lax.while_loop(cond, body, init_state)

    key_v, _ = random.split(key)
    v  = random.normal(key_v, shape=(n_samples, *batch_shape, d))
    mu_exp = mu[None, ...]                                # (1, ..., d)

    v = v - (v * mu_exp).sum(axis=-1, keepdims=True) * mu_exp
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)

    w_final = w_final[..., None]                          # (n, ..., 1)
    x = w_final * mu_exp + jnp.sqrt(1.0 - w_final**2) * v

    return x #.astype(mu.dtype)
