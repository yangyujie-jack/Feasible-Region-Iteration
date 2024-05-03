import jax.numpy as jnp


def mask_average(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1)
