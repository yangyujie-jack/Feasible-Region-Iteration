from typing import NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from relax.algorithm.base import Algorithm
from relax.network.pb import VBLNet, PBParams
from relax.utils.experience import Experience
from relax.utils.jax_utils import mask_average
from relax.utils.typing import Metric


class PBOptStates(NamedTuple):
    model: optax.OptState
    policy: optax.OptState
    barrier: optax.OptState


class PBTrainState(NamedTuple):
    params: PBParams
    opt_state: PBOptStates


class VBL(Algorithm):
    def __init__(self, agent: VBLNet, params: PBParams, *, lr: float = 3e-4, lam: float = 0.1, eps: float = 0.01):
        self.agent = agent
        self.lam = lam
        self.eps = eps

        self.optim = optax.adam(lr)

        self.state = PBTrainState(
            params=params,
            opt_state=PBOptStates(
                model=self.optim.init(params.model),
                policy=self.optim.init(params.policy),
                barrier=self.optim.init(params.barrier),
            ),
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray, state: PBTrainState, data: Experience
        ) -> Tuple[PBTrainState, Metric]:
            obs, action, next_obs, feasible, infeasible = (
                data.obs,
                data.action,
                data.next_obs,
                data.feasible,
                data.infeasible,
            )
            model_params, policy_params, barrier_params = state.params
            model_opt_state, policy_opt_state, barrier_opt_state = state.opt_state
            del key

            obs = self.agent.preprocess(obs)
            next_obs = self.agent.preprocess(next_obs)

            # update model
            def model_loss_fn(model_params: hk.Params) -> jnp.ndarray:
                next_obs_pred = obs + self.agent.model(model_params, obs, action)
                model_loss = jnp.mean((next_obs - next_obs_pred) ** 2)
                return model_loss

            model_loss, model_grads = jax.value_and_grad(model_loss_fn)(model_params)
            model_updates, model_opt_state = self.optim.update(model_grads, model_opt_state)
            model_params = optax.apply_updates(model_params, model_updates)

            # update barrier
            new_action = self.agent.evaluate(policy_params, obs)
            new_next_obs = obs + self.agent.model(model_params, obs, new_action)

            def barrier_loss_fn(barrier_params: hk.Params) -> jnp.ndarray:
                barrier = self.agent.barrier(barrier_params, obs)
                next_barrier = self.agent.barrier(barrier_params, new_next_obs)
                feasible_loss = mask_average(jnp.maximum(self.eps + barrier, 0), feasible)
                infeasible_loss = mask_average(jnp.maximum(self.eps - barrier, 0), infeasible)
                invariant_loss = jnp.maximum(self.eps + next_barrier - (1 - self.lam) * barrier, 0).mean()
                barrier_loss = feasible_loss + infeasible_loss + invariant_loss
                return barrier_loss

            barrier_loss, barrier_grads = jax.value_and_grad(barrier_loss_fn)(barrier_params)
            barrier_updates, barrier_opt_state = self.optim.update(barrier_grads, barrier_opt_state)
            barrier_params = optax.apply_updates(barrier_params, barrier_updates)

            # update policy
            def policy_loss_fn(policy_params: hk.Params) -> jnp.ndarray:
                new_action = self.agent.evaluate(policy_params, obs)
                new_next_obs = obs + self.agent.model(model_params, obs, new_action)
                barrier = self.agent.barrier(barrier_params, obs)
                next_barrier = self.agent.barrier(barrier_params, new_next_obs)
                policy_loss = jnp.maximum(self.eps + next_barrier - (1 - self.lam) * barrier, 0).mean()
                return policy_loss

            policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
            policy_updates, policy_opt_state = self.optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_updates)

            state = PBTrainState(
                params=PBParams(model_params, policy_params, barrier_params),
                opt_state=PBOptStates(model_opt_state, policy_opt_state, barrier_opt_state),
            )

            info = {
                "model_loss": model_loss,
                "barrier_loss": barrier_loss,
                "label_feasible_ratio": feasible.mean(),
                "label_infeasible_ratio": infeasible.mean(),
                "policy_loss": policy_loss,
            }

            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
