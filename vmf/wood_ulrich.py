import jax
import jax.numpy as jnp
from jax import lax, random
from functools import partial

def _beta(key, a, b, shape):
    """Draw Beta(a,b) via two Gammas – works on all back‑ends."""
    k1, k2 = random.split(key)
    gamma1 = random.gamma(k1, a, shape)
    gamma2 = random.gamma(k2, b, shape)
    return gamma1 / (gamma1 + gamma2)

def _while_vmf_wood_cond(v):
    """Condition for while loop: continue until all samples accepted."""
    # return jnp.logical_not( jnp.all(v[2]) ) # v[2] is the mask of unaccepted samples
    return jnp.logical_not( jnp.any(v[2]) ) # v[2] is the mask of unaccepted samples

@partial(jax.jit, static_argnums=(3,))
def sample_vmf_wood(key, mu, kappa, n_samples):
    """
    Wood‑Ulrich vMF sampler, vectorised over leading batch dims.

    Args
    ----
    key       : PRNGKey
    mu        : (..., d) array, *need not* be unit‑norm (we'll normalise)
    kappa     : (...,)   array with kappa >= 0
    n_samples : int      number of samples to draw (per batch element)

    Returns
    -------
    x         : (n_samples, ..., d) array of vMF samples on S^{d-1}
    """
    mu = mu / jnp.linalg.norm(mu, axis=-1, keepdims=True)
    *batch, d = mu.shape
    batch_shape = tuple(batch)

    kappa = jnp.broadcast_to(kappa, batch_shape)

    # Wood's constants (vectorised)
    b = (-2. * kappa + jnp.sqrt(4. * kappa**2 + (d - 1.)**2)) / (d - 1.)
    x0 = (1. - b) / (1. + b)
    c  = kappa * x0 + (d - 1.) * jnp.log1p(-x0**2)

    def body_fun(state):
        key, w, mask = state
        key, sub = random.split(key)

        # draw w proposal from Beta((d-1)/2, (d-1)/2)
        a = (d - 1.) / 2.
        z = _beta(sub, a, a, shape=(n_samples, *batch_shape))
        w_prop = (1. - (1. + b) * z) / (1. - (1. - b) * z)

        # accept / reject
        key, sub = random.split(key)
        u = random.uniform(sub, shape=(n_samples, *batch_shape))
        accept = (
            kappa * w_prop
            + (d - 1.) * jnp.log1p(-x0 * w_prop)
            - c
            >= jnp.log(u)
        )

        # update where still missing
        w = jnp.where(accept & mask, w_prop, w)
        mask = jnp.where(accept, False, mask)
        return (key, w, mask)

    # initialise loop state
    init_key, sub = random.split(key)
    w_init = jnp.zeros((n_samples, *batch_shape))
    mask_init = jnp.full((n_samples, *batch_shape), True)

    ### TODO: Fix this loop. Is broke
    # keep sampling until all n_samples accepted
    # key_out, w_final, _ = lax.while_loop(
    #     lambda s: jnp.any(s[2]),
    #     body_fun,
    #     (sub, w_init, mask_init)
    # )
    ### Works just fine
    # key_out, w_final, _ = body_fun((sub, w_init, mask_init))

    key_out, w_final, _ = lax.while_loop(
        cond_fun=_while_vmf_wood_cond,
        body_fun=body_fun,
        init_val=(sub, w_init, mask_init),
    )

    # tangent direction V: sample Gaussian, project & normalise
    key_V, _ = random.split(key_out)
    v = random.normal(key_V, shape=(n_samples, *batch_shape, d))
    # remove component along mu
    v = v - (v * mu).sum(axis=-1, keepdims=True) * mu
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)

    w_final = w_final[..., None]  # shape (n, ..., 1)
    x = w_final * mu + jnp.sqrt(1. - w_final**2) * v
    return x  # (n_samples, ..., d)


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
