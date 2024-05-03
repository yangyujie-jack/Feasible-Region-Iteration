from typing import NamedTuple, Optional

import numpy as np
import jax.numpy as jnp


class Experience(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    cost: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray
    feasible: jnp.ndarray
    infeasible: jnp.ndarray
    barrier: jnp.ndarray
    next_barrier: jnp.ndarray

    def batch_size(self) -> Optional[int]:
        try:
            if self.reward.ndim > 0:
                return self.reward.shape[0]
            else:
                return None
        except AttributeError:
            return None

    def __repr__(self):
        return f"Experience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return Experience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            cost=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool8),
            feasible=np.zeros(leading_dims, dtype=np.bool8),
            infeasible=np.zeros(leading_dims, dtype=np.bool8),
            barrier=np.zeros(leading_dims, dtype=np.float32),
            next_barrier=np.zeros(leading_dims, dtype=np.float32),
        )
