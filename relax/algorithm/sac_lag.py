from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.sac_lag import SACLagNet, SACLagParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class SACLagOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    cost_q: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    multiplier: optax.OptState


class SACLagTrainState(NamedTuple):
    params: SACLagParams
    opt_state: SACLagOptStates
    step: int


class SACLag(Algorithm):
    def __init__(
        self,
        agent: SACLagNet,
        params: SACLagParams,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        multiplier_lr: float = 3e-4,
        multiplier_delay: int = 10,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.multiplier_delay = multiplier_delay

        self.optim = optax.adam(lr)
        self.multiplier_optim = optax.adam(multiplier_lr)

        self.state = SACLagTrainState(
            params=params,
            opt_state=SACLagOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                cost_q=self.optim.init(params.cost_q),
                policy=self.optim.init(params.policy),
                log_alpha=self.optim.init(params.log_alpha),
                multiplier=self.multiplier_optim.init(params.multiplier_param),
            ),
            step=0,
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray, state: SACLagTrainState, data: Experience
        ) -> Tuple[SACLagTrainState, Metric]:
            obs, action, reward, cost, next_obs, done = (
                data.obs,
                data.action,
                data.reward,
                data.cost,
                data.next_obs,
                data.done,
            )
            (
                q1_params,
                q2_params,
                target_q1_params,
                target_q2_params,
                cost_q_params,
                target_cost_q_params,
                policy_params,
                log_alpha,
                multiplier_param,
            ) = state.params
            (
                q1_opt_state,
                q2_opt_state,
                cost_q_opt_state,
                policy_opt_state,
                log_alpha_opt_state,
                multiplier_opt_state,
            ) = state.opt_state
            step = state.step

            next_eval_key, new_eval_key = jax.random.split(key)

            # compute target q & cost q
            next_action, next_logp = self.agent.evaluate(next_eval_key, policy_params, next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target
            cost_q_target = self.agent.q(target_cost_q_params, next_obs, next_action)
            cost_q_backup = cost + (1 - done) * self.gamma * cost_q_target

            # update q
            def q_loss_fn(q_params: hk.Params) -> jnp.ndarray:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss

            q1_loss, q1_grads = jax.value_and_grad(q_loss_fn)(q1_params)
            q2_loss, q2_grads = jax.value_and_grad(q_loss_fn)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # update cost q
            def cost_q_loss_fn(cost_q_params: hk.Params) -> jnp.ndarray:
                cost_q = self.agent.q(cost_q_params, obs, action)
                cost_q_loss = jnp.mean((cost_q - cost_q_backup) ** 2)
                return cost_q_loss

            cost_q_loss, cost_q_grads = jax.value_and_grad(cost_q_loss_fn)(cost_q_params)
            cost_q_update, cost_q_opt_state = self.optim.update(cost_q_grads, cost_q_opt_state)
            cost_q_params = optax.apply_updates(cost_q_params, cost_q_update)

            # update policy
            def policy_loss_fn(policy_params: hk.Params) -> jnp.ndarray:
                new_action, new_logp = self.agent.evaluate(new_eval_key, policy_params, obs)
                q1 = self.agent.q(q1_params, obs, new_action)
                q2 = self.agent.q(q2_params, obs, new_action)
                q = jnp.minimum(q1, q2)
                cost_q = self.agent.q(cost_q_params, obs, new_action)
                policy_loss = jnp.mean(jnp.exp(log_alpha) * new_logp - q + jax.nn.softplus(multiplier_param) * cost_q)
                return policy_loss, (q1, q2, cost_q, new_logp)

            (policy_loss, aux), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)
            q1, q2, cost_q, new_logp = aux
            policy_update, policy_opt_state = self.optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jnp.ndarray) -> jnp.ndarray:
                log_alpha_loss = -jnp.mean(log_alpha * (new_logp + self.agent.target_entropy))
                return log_alpha_loss

            log_alpha_grads = jax.grad(log_alpha_loss_fn)(log_alpha)
            log_alpha_update, log_alpha_opt_state = self.optim.update(log_alpha_grads, log_alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, log_alpha_update)

            # update multiplier
            def multiplier_loss_fn(multiplier_param: jnp.ndarray) -> jnp.ndarray:
                multiplier_loss = -jnp.mean(multiplier_param * cost_q)
                return multiplier_loss

            def multiplier_update_fn(
                multiplier_param: jnp.ndarray, multiplier_opt_state: optax.OptState
            ) -> Tuple[jnp.ndarray, optax.OptState]:
                multiplier_grads = jax.grad(multiplier_loss_fn)(multiplier_param)
                multiplier_update, multiplier_opt_state = self.multiplier_optim.update(
                    multiplier_grads, multiplier_opt_state
                )
                multiplier_param = optax.apply_updates(multiplier_param, multiplier_update)
                return multiplier_param, multiplier_opt_state

            multiplier_param, multiplier_opt_state = jax.lax.cond(
                step % self.multiplier_delay == 0,
                multiplier_update_fn,
                lambda *x: x,
                multiplier_param,
                multiplier_opt_state,
            )

            # update target q & cost q
            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)
            target_cost_q_params = optax.incremental_update(cost_q_params, target_cost_q_params, self.tau)

            state = SACLagTrainState(
                params=SACLagParams(
                    q1=q1_params,
                    q2=q2_params,
                    target_q1=target_q1_params,
                    target_q2=target_q2_params,
                    cost_q=cost_q_params,
                    target_cost_q=target_cost_q_params,
                    policy=policy_params,
                    log_alpha=log_alpha,
                    multiplier_param=multiplier_param,
                ),
                opt_state=SACLagOptStates(
                    q1=q1_opt_state,
                    q2=q2_opt_state,
                    cost_q=cost_q_opt_state,
                    policy=policy_opt_state,
                    log_alpha=log_alpha_opt_state,
                    multiplier=multiplier_opt_state,
                ),
                step=step + 1,
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "cost_q_loss": cost_q_loss,
                "cost_q": jnp.mean(cost_q),
                "policy_loss": policy_loss,
                "entropy": -jnp.mean(new_logp),
                "alpha": jnp.exp(log_alpha),
                "multiplier": jax.nn.softplus(multiplier_param),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
