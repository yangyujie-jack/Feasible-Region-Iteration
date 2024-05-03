from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.fsi import FSINet, FSIParams
from relax.utils.experience import Experience
from relax.utils.jax_utils import mask_average
from relax.utils.typing import Metric


class FSIOptStates(NamedTuple):
    model: optax.OptState
    policy: optax.OptState
    barrier: optax.OptState
    classifier: optax.OptState


class FSITrainState(NamedTuple):
    params: FSIParams
    opt_state: FSIOptStates


class FSI(Algorithm):
    def __init__(
        self,
        agent: FSINet,
        params: FSIParams,
        *,
        gamma: float = 0.999,
        lr: float = 3e-4,
        tau: float = 0.005,
        feasible_threshold: float = 0.1,
        infeasible_threshold: float = 0.9,
        lam: float = 0.1,
        eps: float = 0.01,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.feasible_threshold = feasible_threshold
        self.infeasible_threshold = infeasible_threshold
        self.lam = lam
        self.eps = eps

        self.optim = optax.adam(lr)

        self.state = FSITrainState(
            params=params,
            opt_state=FSIOptStates(
                model=self.optim.init(params.model),
                policy=self.optim.init(params.policy),
                barrier=self.optim.init(params.barrier),
                classifier=self.optim.init(params.classifier),
            ),
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray, state: FSITrainState, data: Experience
        ) -> Tuple[FSITrainState, Metric]:
            obs, action, next_obs, feasible, infeasible = (
                data.obs,
                data.action,
                data.next_obs,
                data.feasible,
                data.infeasible,
            )
            model_params, policy_params, barrier_params, classifier_params, target_classifier_params = state.params
            model_opt_state, policy_opt_state, barrier_opt_state, classifier_opt_state = state.opt_state
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

            # update classifier
            new_action = self.agent.evaluate(policy_params, obs)
            new_next_obs = obs + self.agent.model(model_params, obs, new_action)

            def classifier_loss_fn(classifier_params: hk.Params):
                logits = self.agent.classifier(classifier_params, obs)
                labeled_target = 0 * feasible + 1 * infeasible
                labeled = feasible + infeasible
                supervised_loss = mask_average(optax.sigmoid_binary_cross_entropy(logits, labeled_target), labeled)
                next_probs = jax.nn.sigmoid(self.agent.classifier(target_classifier_params, new_next_obs))
                unlabeled_target = self.gamma * jax.lax.stop_gradient(next_probs)
                unsupervised_loss = mask_average(
                    optax.sigmoid_binary_cross_entropy(logits, unlabeled_target), 1 - labeled
                )
                classifier_loss = supervised_loss + unsupervised_loss
                return classifier_loss, (logits,)

            (classifier_loss, aux), classifier_grads = jax.value_and_grad(classifier_loss_fn, has_aux=True)(
                classifier_params
            )
            (logits,) = aux
            classifier_updates, classifier_opt_state = self.optim.update(classifier_grads, classifier_opt_state)
            classifier_params = optax.apply_updates(classifier_params, classifier_updates)

            # update barrier
            def barrier_loss_fn(barrier_params: hk.Params):
                barrier = self.agent.barrier(barrier_params, obs)
                next_barrier = self.agent.barrier(barrier_params, new_next_obs)
                probs = jax.nn.sigmoid(logits)
                classifier_feasible = probs < self.feasible_threshold
                classifier_infeasible = probs > self.infeasible_threshold
                feasible_loss = mask_average(jnp.maximum(self.eps + barrier, 0), classifier_feasible)
                infeasible_loss = mask_average(jnp.maximum(self.eps - barrier, 0), classifier_infeasible)
                invariant_loss = jnp.maximum(self.eps + next_barrier - (1 - self.lam) * barrier, 0).mean()
                barrier_loss = feasible_loss + infeasible_loss + invariant_loss
                return barrier_loss, (
                    feasible_loss,
                    infeasible_loss,
                    invariant_loss,
                    classifier_feasible,
                    classifier_infeasible,
                )

            (barrier_loss, aux), barrier_grads = jax.value_and_grad(barrier_loss_fn, has_aux=True)(barrier_params)
            feasible_loss, infeasible_loss, invariant_loss, classifier_feasible, classifier_infeasible = aux
            barrier_updates, barrier_opt_state = self.optim.update(barrier_grads, barrier_opt_state)
            barrier_params = optax.apply_updates(barrier_params, barrier_updates)

            # update policy
            def policy_loss_fn(policy_params: hk.Params) -> jnp.ndarray:
                new_action = self.agent.evaluate(policy_params, obs)
                new_next_obs = obs + self.agent.model(model_params, obs, new_action)
                next_probs = jax.nn.sigmoid(self.agent.classifier(classifier_params, new_next_obs))
                policy_loss = next_probs.mean()
                return policy_loss

            policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
            policy_updates, policy_opt_state = self.optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_updates)

            # update target classifier
            target_classifier_params = optax.incremental_update(classifier_params, target_classifier_params, self.tau)

            state = FSITrainState(
                params=FSIParams(
                    model_params, policy_params, barrier_params, classifier_params, target_classifier_params
                ),
                opt_state=FSIOptStates(model_opt_state, policy_opt_state, barrier_opt_state, classifier_opt_state),
            )

            info = {
                "model_loss": model_loss,
                "classifier_loss": classifier_loss,
                "feasible_loss": feasible_loss,
                "infeasible_loss": infeasible_loss,
                "invariant_loss": invariant_loss,
                "barrier_loss": barrier_loss,
                "label_feasible_ratio": feasible.mean(),
                "label_infeasible_ratio": infeasible.mean(),
                "classifier_feasible_ratio": classifier_feasible.mean(),
                "classifier_infeasible_ratio": classifier_infeasible.mean(),
                "policy_loss": policy_loss,
            }

            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
