from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, ValueNet, ModelNet, DeterministicPolicyNet
from relax.network.common import WithSquashedDeterministicPolicy


class FSIParams(NamedTuple):
    model: hk.Params
    policy: hk.Params
    barrier: hk.Params
    classifier: hk.Params
    target_classifier: hk.Params


@dataclass
class FSINet(WithSquashedDeterministicPolicy):
    model: Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    barrier: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    classifier: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]


def create_fsi_net(
    key: jax.random.KeyArray,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    preprocess: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    activation: Activation = jax.nn.relu,
    exploration_noise: float = 0.1,
) -> Tuple[FSINet, FSIParams]:
    model = hk.without_apply_rng(hk.transform(lambda obs, act: ModelNet(obs_dim, hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(
        hk.transform(lambda obs: DeterministicPolicyNet(act_dim, hidden_sizes, activation)(obs))
    )
    barrier = hk.without_apply_rng(hk.transform(lambda obs: ValueNet(hidden_sizes, jax.nn.elu)(obs)))
    classifier = hk.without_apply_rng(hk.transform(lambda obs: ValueNet(hidden_sizes, jax.nn.elu)(obs)))

    @jax.jit
    def init(key, obs, act):
        model_key, policy_key, barrier_key, classifier_key = jax.random.split(key, 4)
        model_params = model.init(model_key, obs, act)
        policy_params = policy.init(policy_key, obs)
        barrier_params = barrier.init(barrier_key, obs)
        classifier_params = classifier.init(classifier_key, obs)
        target_classifier_params = classifier_params
        return FSIParams(model_params, policy_params, barrier_params, classifier_params, target_classifier_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = FSINet(
        policy=policy.apply,
        model=model.apply,
        barrier=barrier.apply,
        classifier=classifier.apply,
        preprocess=preprocess,
        exploration_noise=exploration_noise,
    )
    return net, params
