import pickle

import numpy as np
import jax, jax.numpy as jnp

from relax.utils.experience import Experience
from relax.utils.flag import RELAX_USE_CUDA
from relax.utils.typing import Metric


class Algorithm:
    # NOTE: a not elegant blanket implementation of the algorithm interface
    def _implement_common_behavior(self, stateless_update, stateless_get_action, stateless_get_deterministic_action):
        if RELAX_USE_CUDA:
            self._update = jax.jit(stateless_update, donate_argnums=(1,))
        else:
            self._update = jax.jit(stateless_update)  # Donation is not implemented for cpu.
        self._get_action = jax.jit(stateless_get_action)
        self._get_deterministic_action = jax.jit(stateless_get_deterministic_action)

    if RELAX_USE_CUDA:
        # Version for CUDA
        def update(self, key: jax.random.KeyArray, data: Experience) -> Metric:
            # NOTE: Removing this leads to slowdown in other parts of the code (env.step), to be investigated
            data = jax.device_put(data)
            self.state, info = self._update(key, self.state, data)
            for v in info.values():
                v.copy_to_host_async()
            return {k: float(v) for k, v in info.items()}

        def get_action(self, key: jax.random.KeyArray, obs: np.ndarray) -> np.ndarray:
            action = self._get_action(key, self.state.params.policy, obs)
            return np.asarray(action)

        def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
            action = self._get_deterministic_action(self.state.params.policy, obs)
            return np.asarray(action)

    else:
        # Version optimized for CPU
        def update(self, key: jax.random.KeyArray, data: Experience) -> Metric:
            self.state, info = self._update(key, self.state, data)
            return {k: float(v) for k, v in info.items()}

        def get_action(self, key: jax.random.KeyArray, obs: np.ndarray) -> np.ndarray:
            action = self._get_action(key, self.state.params.policy, obs)
            return np.asarray(action)

        def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
            action = self._get_deterministic_action(self.state.params.policy, obs)
            return np.asarray(action)

    def save(self, path: str) -> None:
        state = jax.device_get(self.state)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.state = jax.device_put(state)
