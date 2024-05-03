from typing import NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from relax.algorithm.base import Algorithm
from relax.network.sac_pb import SACVBLNet, SACPBParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class SACPBOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    model: optax.OptState
    safe_policy: optax.OptState
    barrier: optax.OptState
    multiplier: optax.OptState


class SACPBTrainState(NamedTuple):
    params: SACPBParams
    opt_state: SACPBOptStates
    step: int


class SACVBL(Algorithm):
    def __init__(
        self,
        agent: SACVBLNet,
        params: SACPBParams,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        lam: float = 0.1,
        eps: float = 0.01,
        multiplier_lr: float = 3e-4,
        multiplier_delay: int = 10,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.lam = lam
        self.eps = eps
        self.multiplier_delay = multiplier_delay

        self.optim = optax.adam(lr)
        self.multiplier_optim = optax.adam(multiplier_lr)

        self.state = SACPBTrainState(
            params=params,
            opt_state=SACPBOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.optim.init(params.policy),
                log_alpha=self.optim.init(params.log_alpha),
                model=self.optim.init(params.model),
                safe_policy=self.optim.init(params.safe_policy),
                barrier=self.optim.init(params.barrier),
                multiplier=self.multiplier_optim.init(params.multiplier_param),
            ),
            step=0,
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray, state: SACPBTrainState, data: Experience
        ) -> Tuple[SACPBTrainState, Metric]:
            obs, action, reward, next_obs, reward, done, feasible, infeasible = (
                data.obs,
                data.action,
                data.reward,
                data.next_obs,
                data.reward,
                data.done,
                data.feasible,
                data.infeasible,
            )
            (
                q1_params,
                q2_params,
                target_q1_params,
                target_q2_params,
                policy_params,
                log_alpha,
                model_params,
                safe_policy_params,
                barrier_params,
                multiplier_param,
            ) = state.params
            (
                q1_opt_state,
                q2_opt_state,
                policy_opt_state,
                log_alpha_opt_state,
                model_opt_state,
                safe_policy_opt_state,
                barrier_opt_state,
                multiplier_opt_state,
            ) = state.opt_state
            step = state.step
            next_eval_key, new_eval_key = jax.random.split(key)

            obs_preprocess = self.agent.preprocess(obs)
            next_obs_preprocess = self.agent.preprocess(next_obs)

            # update model
            def model_loss_fn(model_params: hk.Params):
                next_obs_pred = obs_preprocess + self.agent.model(model_params, obs_preprocess, action)
                model_loss = jnp.mean((next_obs_preprocess - next_obs_pred) ** 2)
                return model_loss

            model_loss, model_grads = jax.value_and_grad(model_loss_fn)(model_params)
            model_updates, model_opt_state = self.optim.update(model_grads, model_opt_state)
            model_params = optax.apply_updates(model_params, model_updates)

            # update barrier
            def barrier_loss_fn(barrier_params: hk.Params):
                barrier = self.agent.barrier(barrier_params, obs_preprocess)
                new_action = self.agent.safe_policy_evaluate(safe_policy_params, obs_preprocess)
                new_next_obs = obs_preprocess + self.agent.model(model_params, obs_preprocess, new_action)
                next_barrier = self.agent.barrier(barrier_params, new_next_obs)
                feasible_loss = jnp.mean(jnp.maximum(self.eps + barrier, 0) * feasible)
                infeasible_loss = jnp.mean(jnp.maximum(self.eps - barrier, 0) * infeasible)
                invariant_loss = jnp.mean(jnp.maximum(self.eps + next_barrier - (1 - self.lam) * barrier, 0))
                barrier_loss = feasible_loss + infeasible_loss + invariant_loss
                return barrier_loss, (feasible_loss, infeasible_loss, invariant_loss)

            (barrier_loss, aux), barrier_grads = jax.value_and_grad(barrier_loss_fn, has_aux=True)(barrier_params)
            feasible_loss, infeasible_loss, invariant_loss = aux
            barrier_updates, barrier_opt_state = self.optim.update(barrier_grads, barrier_opt_state)
            barrier_params = optax.apply_updates(barrier_params, barrier_updates)

            # update safe policy
            def safe_policy_loss_fn(safe_policy_params: hk.Params):
                barrier = self.agent.barrier(barrier_params, obs_preprocess)
                new_action = self.agent.safe_policy_evaluate(safe_policy_params, obs_preprocess)
                new_next_obs = obs_preprocess + self.agent.model(model_params, obs_preprocess, new_action)
                next_barrier = self.agent.barrier(barrier_params, new_next_obs)
                safe_policy_loss = jnp.mean(jnp.maximum(self.eps + next_barrier - (1 - self.lam) * barrier, 0))
                return safe_policy_loss

            safe_policy_loss, safe_policy_grads = jax.value_and_grad(safe_policy_loss_fn)(safe_policy_params)
            safe_policy_updates, safe_policy_opt_state = self.optim.update(safe_policy_grads, safe_policy_opt_state)
            safe_policy_params = optax.apply_updates(safe_policy_params, safe_policy_updates)

            # update q
            next_action, next_logp = self.agent.evaluate(next_eval_key, policy_params, next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params):
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # update policy
            def policy_loss_fn(policy_params: hk.Params):
                new_action, new_logp = self.agent.evaluate(new_eval_key, policy_params, obs)
                q1 = self.agent.q(q1_params, obs, new_action)
                q2 = self.agent.q(q2_params, obs, new_action)
                q = jnp.minimum(q1, q2)
                barrier = self.agent.barrier(barrier_params, obs_preprocess)
                new_next_obs = obs_preprocess + self.agent.model(model_params, obs_preprocess, new_action)
                next_barrier = self.agent.barrier(barrier_params, new_next_obs)
                barrier_penalty = jnp.maximum(self.eps + next_barrier - (1 - self.lam) * barrier, 0)
                policy_loss = jnp.mean(jnp.exp(log_alpha) * new_logp - q +
                                       jax.nn.softplus(multiplier_param) * barrier_penalty)
                return policy_loss, (new_logp, barrier_penalty)

            (policy_loss, aux), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)
            new_logp, barrier_penalty = aux
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
                multiplier_loss = -jnp.mean(multiplier_param * barrier_penalty)
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

            # update target q
            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)

            state = SACPBTrainState(
                params=SACPBParams(
                    q1_params,
                    q2_params,
                    target_q1_params,
                    target_q2_params,
                    policy_params,
                    log_alpha,
                    model_params,
                    safe_policy_params,
                    barrier_params,
                    multiplier_param,
                ),
                opt_state=SACPBOptStates(
                    q1_opt_state,
                    q2_opt_state,
                    policy_opt_state,
                    log_alpha_opt_state,
                    model_opt_state,
                    safe_policy_opt_state,
                    barrier_opt_state,
                    multiplier_opt_state,
                ),
                step=step + 1,
            )
            info = {
                "model_loss": model_loss,
                "feasible_loss": feasible_loss,
                "infeasible_loss": infeasible_loss,
                "invariant_loss": invariant_loss,
                "barrier_loss": barrier_loss,
                "label_feasible_ratio": jnp.mean(feasible),
                "label_infeasible_ratio": jnp.mean(infeasible),
                "safe_policy_loss": safe_policy_loss,
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "policy_loss": policy_loss,
                "entropy": -jnp.mean(new_logp),
                "alpha": jnp.exp(log_alpha),
                "multiplier": jax.nn.softplus(multiplier_param),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
