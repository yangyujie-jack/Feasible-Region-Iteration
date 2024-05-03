from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import jax
import numpy as np
from tensorboardX import SummaryWriter

from relax.algorithm import Algorithm
from relax.buffer import ExperienceBuffer
from relax.utils.experience import Experience


class OffPolicyTrainer:
    def __init__(
        self,
        env,
        algorithm: Algorithm,
        buffer: ExperienceBuffer,
        log_path: Path,
        batch_size: int = 256,
        start_step: int = 1000,
        total_step: int = int(1e6),
        sample_per_iteration: int = 1,
        update_per_iteration: int = 1,
        evaluate_env: Optional = None,
        evaluate_every: int = 10000,
        evaluate_n_episode: int = 10,
        sample_log_n_episode: int = 10,
        update_log_n_step: int = 1000,
    ):
        self.env = env
        self.algorithm = algorithm
        self.buffer = buffer
        self.batch_size = batch_size
        self.start_step = start_step
        self.total_step = total_step
        self.sample_per_iteration = sample_per_iteration
        self.update_per_iteration = update_per_iteration
        self.log_path = log_path
        self.state_path = self.log_path / "state.pkl"
        self.logger = SummaryWriter(str(self.log_path))
        self.evaluate_env = evaluate_env
        self.evaluate_every = evaluate_every
        self.evaluate_n_episode = evaluate_n_episode
        self.sample_log_n_episode = sample_log_n_episode
        self.update_log_n_step = update_log_n_step

    def train(self, key: jax.random.KeyArray):
        iter_key_fn = create_iter_key_fn(key)
        sample_step, sample_episode, update_step = 0, 0, 0
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        sample_info = {
            "episode_return": [],
            "episode_cost": [],
            "episode_length": [],
        }
        update_info: Dict[str, list] = {}
        obs = self.env.reset()
        while sample_step < self.total_step:
            # setup random keys
            sample_key, update_key = iter_key_fn(sample_step)
            # sample data
            for _ in range(self.sample_per_iteration):
                if sample_step < self.start_step:
                    action = self.env.action_space.sample()
                else:
                    action = self.algorithm.get_action(sample_key, obs)
                next_obs, reward, done, info = self.env.step(action)
                cost = info.get("cost", 0.0)
                feasible = info.get("feasible", False)
                infeasible = info.get("infeasible", False)
                barrier = info.get("barrier", 0.0)
                next_barrier = info.get("next_barrier", 0.0)

                experience = Experience(
                    obs, action, reward, cost, next_obs, done, feasible, infeasible, barrier, next_barrier
                )
                self.buffer.add(experience)

                ep_ret += reward
                ep_cost += cost
                ep_len += 1
                sample_step += 1

                if done:
                    sample_info["episode_return"].append(ep_ret)
                    sample_info["episode_cost"].append(ep_cost)
                    sample_info["episode_length"].append(ep_len)
                    sample_episode += 1

                    if sample_episode % self.sample_log_n_episode == 0:
                        for k, v in sample_info.items():
                            self.logger.add_scalar(f"sample/{k}", np.mean(v), sample_step)
                            sample_info[k] = []
                        print("sample step", sample_step)

                    obs = self.env.reset()
                    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
                else:
                    obs = next_obs

            if sample_step < self.start_step:
                continue

            # update parameters
            for _ in range(self.update_per_iteration):
                data = self.buffer.sample(self.batch_size)
                info = self.algorithm.update(update_key, data)
                for k, v in info.items():
                    if k in update_info:
                        update_info[k].append(v)
                    else:
                        update_info[k] = [v]

                update_step += 1

                if update_step % self.update_log_n_step == 0:
                    for k, v in update_info.items():
                        self.logger.add_scalar(f"update/{k}", np.mean(v), update_step)
                        update_info[k] = []
                    print("update step", update_step)
                    self.algorithm.save(self.state_path)

            # evaluate
            if self.evaluate_env is not None and sample_step % self.evaluate_every == 0:
                self.evaluate(sample_step)

    def evaluate(self, sample_step: int):
        eval_info = {
            "episode_return": [],
            "episode_cost": [],
            "episode_length": [],
        }
        for _ in range(self.evaluate_n_episode):
            eval_ep_ret, eval_ep_cost, eval_ep_len = 0.0, 0.0, 0
            obs = self.evaluate_env.reset()
            while True:
                action = self.algorithm.get_deterministic_action(obs)
                obs, reward, done, info = self.evaluate_env.step(action)
                cost = info.get("cost", 0.0)

                eval_ep_ret += reward
                eval_ep_cost += cost
                eval_ep_len += 1

                if done:
                    eval_info["episode_return"].append(eval_ep_ret)
                    eval_info["episode_cost"].append(eval_ep_cost)
                    eval_info["episode_length"].append(eval_ep_len)
                    break

        for k, v in eval_info.items():
            self.logger.add_scalar(f"evaluate/{k}", np.mean(v), sample_step)


def create_iter_key_fn(key: jax.random.KeyArray) -> Callable[[int], Tuple[jax.random.KeyArray, jax.random.KeyArray]]:
    def iter_key_fn(step: int):
        iter_key = jax.random.fold_in(key, step)
        sample_key, update_key = jax.random.split(iter_key)
        return sample_key, update_key

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up
    return iter_key_fn
