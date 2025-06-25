import jax.numpy as jnp
from jax import lax, random

# --------------------------------------------------------------------
# helper ― Beta(a,b) via Gamma trick (there is no jax.random.beta yet)
# --------------------------------------------------------------------
def _beta(key, a, b, *, shape):
    k1, k2 = random.split(key)
    ga = random.gamma(k1, a, shape=shape)
    gb = random.gamma(k2, b, shape=shape)
    return ga / (ga + gb)

# --------------------------------------------------------------------
# vectorised Wood-Ulrich vMF sampler, fully JIT-able
# --------------------------------------------------------------------
# @partial(jax.jit, static_argnums=(3,))
def sample_vmf_wood_v2(key, mu, kappa, n_samples):
    """
    Wood-Ulrich vMF sampler, vectorised over leading batch dims.

    Parameters
    ----------
    key       : jax.random.PRNGKey
    mu        : (..., d) array   (need not be unit-norm)
    kappa     : (...,)   array   (κ ≥ 0, broadcastable to mu[...,0])
    n_samples : int              number of samples to draw *per* batch element

    Returns
    -------
    x         : (n_samples, ..., d) array of vMF samples on S^{d−1}
    """
    # ----------------------------------------------------------------
    # prep
    # ----------------------------------------------------------------
    mu = mu / jnp.linalg.norm(mu, axis=-1, keepdims=True)      # normalise μ
    *batch, d = mu.shape
    batch_shape = tuple(batch)

    kappa = jnp.broadcast_to(kappa, batch_shape)               # (...,)

    # Wood constants (vectorised over batch dims)
    b = (-2.0 * kappa + jnp.sqrt(4.0 * kappa ** 2 + (d - 1.0) ** 2)) / (d - 1.0)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1.0) * jnp.log1p(-x0 ** 2)


    # ----------------------------------------------------------------
    # rejection sampling for w ∈ [−1,1]
    # ----------------------------------------------------------------
    a_beta = (d - 1.0) / 2.0                                  # Beta(α,α)

    def body_fun(state):
        key, w, mask = state
        key, k_beta, k_u = random.split(key, 3)

        z = _beta(k_beta, a_beta, a_beta,
                  shape=(n_samples, *batch_shape))             # (n, ...,)

        w_prop = (1.0 - (1.0 + b)[None, ...] * z) / (
                  1.0 - (1.0 - b)[None, ...] * z)              # (n, ...,)

        u = random.uniform(k_u, shape=(n_samples, *batch_shape))

        accept = (
            kappa[None, ...] * w_prop
            + (d - 1.0) * jnp.log1p(-x0[None, ...] * w_prop)
            - c[None, ...]
            >= jnp.log(u)
        )

        # write accepted proposals into w; update mask
        w = jnp.where(accept & mask, w_prop, w)
        mask = jnp.where(accept, False, mask)

        return (key, w, mask)

    def cond_fun(state):
        return jnp.any(state[2])                               # still need draws?

    # initialise loop state
    key, k_loop = random.split(key)
    w0 = jnp.zeros((n_samples, *batch_shape))
    mask0 = jnp.ones((n_samples, *batch_shape), dtype=bool)

    _, w_final, _ = lax.while_loop(cond_fun, body_fun, (k_loop, w0, mask0))

    # ----------------------------------------------------------------
    # tangential component
    # ----------------------------------------------------------------
    key_V, _ = random.split(key)
    v = random.normal(key_V, shape=(n_samples, *batch_shape, d))
    mu_expanded = mu[None, ...]                                # (1, ..., d)

    v = v - (v * mu_expanded).sum(axis=-1, keepdims=True) * mu_expanded
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)

    # ----------------------------------------------------------------
    # assemble sample on S^{d−1}
    # ----------------------------------------------------------------
    w_final = w_final[..., None]                               # (n, ..., 1)
    x = w_final * mu_expanded + jnp.sqrt(1.0 - w_final ** 2) * v
    return x.astype(mu.dtype)                                  # (n_samples, ..., d)
