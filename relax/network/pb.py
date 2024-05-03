from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from relax.network.blocks import Activation, ValueNet, ModelNet, DeterministicPolicyNet
from relax.network.common import WithSquashedDeterministicPolicy


class PBParams(NamedTuple):
    model: hk.Params
    policy: hk.Params
    barrier: hk.Params


@dataclass
class VBLNet(WithSquashedDeterministicPolicy):
    model: Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    barrier: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]


def create_vbl_net(
    key: jax.random.KeyArray,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    preprocess: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    activation: Activation = jax.nn.relu,
    exploration_noise: float = 0.1,
) -> Tuple[VBLNet, PBParams]:
    model = hk.without_apply_rng(hk.transform(lambda obs, act: ModelNet(obs_dim, hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(
        hk.transform(lambda obs: DeterministicPolicyNet(act_dim, hidden_sizes, activation)(obs))
    )
    barrier = hk.without_apply_rng(hk.transform(lambda obs: ValueNet(hidden_sizes, jax.nn.elu)(obs)))

    @jax.jit
    def init(key, obs, act):
        model_key, policy_key, barrier_key = jax.random.split(key, 3)
        model_params = model.init(model_key, obs, act)
        policy_params = policy.init(policy_key, obs)
        barrier_params = barrier.init(barrier_key, obs)
        return PBParams(model_params, policy_params, barrier_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = VBLNet(
        policy=policy.apply,
        model=model.apply,
        barrier=barrier.apply,
        preprocess=preprocess,
        exploration_noise=exploration_noise,
    )
    return net, params
