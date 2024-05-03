import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from relax.network.blocks import Activation, QNet, PolicyNet
from relax.network.common import WithSquashedGaussianPolicy


class SACLagParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    cost_q: hk.Params
    target_cost_q: hk.Params
    policy: hk.Params
    log_alpha: jnp.ndarray
    multiplier_param: jnp.ndarray


@dataclass
class SACLagNet(WithSquashedGaussianPolicy):
    q: Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    target_entropy: float


def create_sac_lag_net(
    key: jax.random.KeyArray,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
) -> Tuple[SACLagNet, SACLagParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs: PolicyNet(act_dim, hidden_sizes, activation)(obs)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, cost_q_key, policy_key = jax.random.split(key, 4)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        cost_q_params = q.init(cost_q_key, obs, act)
        target_cost_q_params = cost_q_params
        policy_params = policy.init(policy_key, obs)
        log_alpha = jnp.array(0.0, dtype=jnp.float32)
        multiplier_param = jnp.array(math.log(math.exp(1) - 1), dtype=jnp.float32)
        return SACLagParams(
            q1=q1_params,
            q2=q2_params,
            target_q1=target_q1_params,
            target_q2=target_q2_params,
            cost_q=cost_q_params,
            target_cost_q=target_cost_q_params,
            policy=policy_params,
            log_alpha=log_alpha,
            multiplier_param=multiplier_param,
        )

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = SACLagNet(policy=policy.apply, q=q.apply, target_entropy=-act_dim)
    return net, params
