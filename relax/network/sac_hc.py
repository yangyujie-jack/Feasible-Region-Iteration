import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from relax.network.blocks import Activation, QNet, PolicyNet, ModelNet
from relax.network.common import WithSquashedGaussianPolicy


class SACHCParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    log_alpha: jnp.ndarray
    model: hk.Params
    multiplier_param: jnp.ndarray


@dataclass
class SACHCNet(WithSquashedGaussianPolicy):
    q: Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    model: Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    target_entropy: float
    barrier: Callable[[jnp.ndarray], jnp.ndarray]
    preprocess: Callable[[jnp.ndarray], jnp.ndarray]


def create_sac_hc_net(
    key: jax.random.KeyArray,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    barrier_input_dim: int,
    barrier: Callable[[jnp.ndarray], jnp.ndarray],
    preprocess: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    activation: Activation = jax.nn.relu,
) -> Tuple[SACHCNet, SACHCParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs: PolicyNet(act_dim, hidden_sizes, activation)(obs)))
    model = hk.without_apply_rng(
        hk.transform(lambda obs, act: ModelNet(barrier_input_dim, hidden_sizes, activation)(obs, act))
    )

    @jax.jit
    def init(key, obs, act, barrier_obs):
        q1_key, q2_key, policy_key, model_key = jax.random.split(key, 4)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs)
        log_alpha = jnp.array(0.0, dtype=jnp.float32)
        model_params = model.init(model_key, barrier_obs, act)
        multiplier_param = jnp.array(math.log(math.exp(1) - 1), dtype=jnp.float32)
        return SACHCParams(
            q1_params,
            q2_params,
            target_q1_params,
            target_q2_params,
            policy_params,
            log_alpha,
            model_params,
            multiplier_param,
        )

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_barrier_obs = jnp.zeros((1, barrier_input_dim))
    params = init(key, sample_obs, sample_act, sample_barrier_obs)

    net = SACHCNet(
        policy=policy.apply,
        q=q.apply,
        model=model.apply,
        target_entropy=-act_dim,
        barrier=barrier,
        preprocess=preprocess,
    )
    return net, params
