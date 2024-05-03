import argparse
import time

import gym
import jax
from safe_env.register import register

from relax.algorithm.fsi import FSI
from relax.algorithm.pb import VBL
from relax.buffer import TreeBuffer
from relax.network.fsi import create_fsi_net
from relax.network.pb import create_vbl_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.utils.experience import Experience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.timing import catchtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="fsi")
    parser.add_argument("--env", type=str, default="DoubleIntegrator-v0")
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--total_step", type=int, default=int(1e4))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--feasible_threshold", type=float, default=0.1)
    parser.add_argument("--infeasible_threshold", type=float, default=0.9)
    parser.add_argument("--sample_log_n_episode", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    register()

    # Manage seeds
    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, buffer_seed, init_network_seed, train_seed = map(int, master_rng.integers(0, 2 ** 32 - 1, 4))
    init_network_key = jax.random.PRNGKey(init_network_seed)
    train_key = jax.random.PRNGKey(train_seed)
    del init_network_seed, train_seed

    env = gym.make(args.env)
    env.seed(env_seed)

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    input_dim = env.barrier_input_dim
    hidden_sizes = [args.hidden_dim] * args.hidden_num
    preprocess = env.preprocess

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=args.total_step, seed=buffer_seed)

    if args.alg == "fsi":
        agent, params = create_fsi_net(init_network_key, input_dim, act_dim, hidden_sizes, preprocess)
        algorithm = FSI(
            agent,
            params,
            lr=args.lr,
            feasible_threshold=args.feasible_threshold,
            infeasible_threshold=args.infeasible_threshold,
        )
    elif args.alg == "vbl":
        agent, params = create_vbl_net(init_network_key, input_dim, act_dim, hidden_sizes, preprocess)
        algorithm = VBL(agent, params, lr=args.lr)
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        total_step=args.total_step,
        log_path=PROJECT_ROOT / "logs" / args.env /
                 (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}'),
        sample_log_n_episode=args.sample_log_n_episode,
    )


    # Warmup jit for more consistent timing
    @catchtime("warmup")
    def warmup_jit():
        dummy_key = jax.random.PRNGKey(0)
        dummy_data = jax.device_put(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
        dummy_state = jax.tree_util.tree_map(jax.numpy.copy, algorithm.state)
        dummy_obs = env.observation_space.sample()
        dummy_state, _ = algorithm._update(dummy_key, dummy_state, dummy_data)
        algorithm._get_action(dummy_key, dummy_state.params.policy, dummy_obs)
        algorithm._get_deterministic_action(dummy_state.params.policy, dummy_obs)


    warmup_jit()

    trainer.train(train_key)
