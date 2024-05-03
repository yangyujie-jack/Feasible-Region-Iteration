from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from relax.network.blocks import Activation, QNet, PolicyNet, ValueNet
from relax.network.common import WithSquashedGaussianPolicy


class FACParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    cost_q: hk.Params
    target_cost_q: hk.Params
    policy: hk.Params
    multiplier: hk.Params
    log_alpha: jnp.ndarray


@dataclass
class FACNet(WithSquashedGaussianPolicy):
    q: Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    multiplier: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    target_entropy: float


def create_fac_net(
    key: jax.random.KeyArray,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
) -> Tuple[FACNet, FACParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs: PolicyNet(act_dim, hidden_sizes, activation)(obs)))
    multiplier = hk.without_apply_rng(hk.transform(
        lambda obs: ValueNet(hidden_sizes, activation, jax.nn.softplus)(obs)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, cost_q_key, policy_key, multiplier_key = jax.random.split(key, 5)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        cost_q_params = q.init(cost_q_key, obs, act)
        target_cost_q_params = cost_q_params
        policy_params = policy.init(policy_key, obs)
        multiplier_params = multiplier.init(multiplier_key, obs)
        log_alpha = jnp.array(0.0, dtype=jnp.float32)
        return FACParams(
            q1=q1_params,
            q2=q2_params,
            target_q1=target_q1_params,
            target_q2=target_q2_params,
            cost_q=cost_q_params,
            target_cost_q=target_cost_q_params,
            policy=policy_params,
            multiplier=multiplier_params,
            log_alpha=log_alpha,
        )

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = FACNet(
        policy=policy.apply,
        q=q.apply,
        multiplier=multiplier.apply,
        target_entropy=-act_dim,
    )
    return net, params
