import os

import jax

RELAX_USE_CUDA = os.environ.get("RELAX_USE_CUDA", "1") == "1"
if RELAX_USE_CUDA:
    default_backend = jax.default_backend()
    if default_backend != "gpu":
        print(f"WARNING: RELAX_USE_CUDA is set to 1 but JAX backend is {default_backend}!")
        RELAX_USE_CUDA = False

if not RELAX_USE_CUDA:
    jax.config.update("jax_platform_name", "cpu")

print(f"Relax is using {'CUDA' if RELAX_USE_CUDA else 'CPU'}.")

# jax.config.update("jax_log_compiles", True)
