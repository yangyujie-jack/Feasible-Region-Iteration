import math
from dataclasses import dataclass
from typing import Callable, Tuple

import jax, jax.numpy as jnp
import haiku as hk
from numpyro.distributions import Normal


@dataclass
class WithSquashedGaussianPolicy:
    policy: Callable[[hk.Params, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]

    def get_action(self, key: jax.random.KeyArray, policy_params: hk.Params, obs: jnp.ndarray) -> jnp.ndarray:
        """for data collection"""
        mean, std = self.policy(policy_params, obs)
        z = Normal(mean, std).sample(key)
        act = jnp.tanh(z)
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jnp.ndarray) -> jnp.ndarray:
        """for evaluation"""
        mean, _ = self.policy(policy_params, obs)
        act = jnp.tanh(mean)
        return act

    def evaluate(
        self, key: jax.random.KeyArray, policy_params: hk.Params, obs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """for algorithm update"""
        mean, std = self.policy(policy_params, obs)
        dist = Normal(mean, std)
        z = dist.rsample(key)
        act = jnp.tanh(z)
        logp = (dist.log_prob(z) - 2 * (math.log(2) - z - jax.nn.softplus(-2 * z))).sum(axis=-1)
        return act, logp


@dataclass
class WithSquashedDeterministicPolicy:
    policy: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    preprocess: Callable[[jnp.ndarray], jnp.ndarray] # TODO: more general?
    exploration_noise: float

    def get_action(self, key: jax.random.KeyArray, policy_params: hk.Params, obs: jnp.ndarray) -> jnp.ndarray:
        """for data collection"""
        obs = self.preprocess(obs)
        z = self.policy(policy_params, obs)
        act = jnp.tanh(z)
        noise = jax.random.normal(key, act.shape) * self.exploration_noise
        act = jnp.clip(act + noise, -1, 1)
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jnp.ndarray) -> jnp.ndarray:
        """for evaluation"""
        obs = self.preprocess(obs)
        z = self.policy(policy_params, obs)
        act = jnp.tanh(z)
        return act

    def evaluate(self, policy_params: hk.Params, processed_obs: jnp.ndarray) -> jnp.ndarray:
        """for algorithm update"""
        z = self.policy(policy_params, processed_obs)
        act = jnp.tanh(z)
        return act
