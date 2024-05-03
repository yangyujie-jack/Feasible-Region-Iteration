import argparse
import time

import gym
import jax
from safe_env.register import register

from relax.algorithm.fac import FAC
from relax.algorithm.sac import SAC
from relax.algorithm.sac_fsi import SACFSI
from relax.algorithm.sac_hc import SACHC
from relax.algorithm.sac_hjr import SACHJR
from relax.algorithm.sac_lag import SACLag
from relax.algorithm.sac_pb import SACVBL
from relax.buffer import TreeBuffer
from relax.network.fac import create_fac_net
from relax.network.sac import create_sac_net
from relax.network.sac_fsi import create_sac_fsi_net
from relax.network.sac_hc import create_sac_hc_net
from relax.network.sac_hjr import create_sac_hjr_net
from relax.network.sac_lag import create_sac_lag_net
from relax.network.sac_pb import create_sac_vbl_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.utils.experience import Experience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.timing import catchtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="sac")
    parser.add_argument("--env", type=str, default="PointGoal-v0")
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(1e4))
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--certificate_lr", type=float, default=3e-4)
    parser.add_argument("--multiplier_lr", type=float, default=3e-4)
    parser.add_argument("--feasible_threshold", type=float, default=0.1)
    parser.add_argument("--infeasible_threshold", type=float, default=0.9)
    parser.add_argument("--penalty_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    register()

    # Manage seeds
    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2 ** 32 - 1, 5)
    )
    init_network_key = jax.random.PRNGKey(init_network_seed)
    train_key = jax.random.PRNGKey(train_seed)
    del init_network_seed, train_seed

    env = gym.make(args.env)
    env.seed(env_seed)

    eval_env = gym.make(args.env)
    eval_env.seed(eval_env_seed)

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    hidden_sizes = [args.hidden_dim] * args.hidden_num
    barrier_input_dim = env.barrier_input_dim
    preprocess = env.preprocess

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=args.total_step, seed=buffer_seed)

    if args.alg == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes)
        algorithm = SAC(agent, params, lr=args.lr)
    elif args.alg == "sac-lag":
        agent, params = create_sac_lag_net(init_network_key, obs_dim, act_dim, hidden_sizes)
        algorithm = SACLag(agent, params, lr=args.lr)
    elif args.alg == "fac":
        agent, params = create_fac_net(init_network_key, obs_dim, act_dim, hidden_sizes)
        algorithm = FAC(agent, params, lr=args.lr)
    elif args.alg == "sac-fsi":
        agent, params = create_sac_fsi_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            barrier_input_dim=barrier_input_dim,
            preprocess=preprocess,
        )
        algorithm = SACFSI(
            agent,
            params,
            lr=args.lr,
            certificate_lr=args.certificate_lr,
            feasible_threshold=args.feasible_threshold,
            infeasible_threshold=args.infeasible_threshold,
            multiplier_lr=args.multiplier_lr,
        )
    elif args.alg == "sac-hc":
        agent, params = create_sac_hc_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            barrier_input_dim=barrier_input_dim,
            barrier=env.handcraft_barrier,
            preprocess=preprocess,
        )
        algorithm = SACHC(agent, params, lr=args.lr)
    elif args.alg == "sac-vbl":
        agent, params = create_sac_vbl_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            barrier_input_dim=barrier_input_dim,
            preprocess=preprocess,
        )
        algorithm = SACVBL(agent, params, lr=args.lr)
    elif args.alg == "sac-hjr":
        agent, params = create_sac_hjr_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            barrier_input_dim=barrier_input_dim,
            preprocess=preprocess,
        )
        algorithm = SACHJR(
            agent,
            params,
            lr=args.lr,
            certificate_lr=args.certificate_lr,
            penalty_scale=args.penalty_scale,
        )
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        evaluate_env=eval_env,
        log_path=PROJECT_ROOT / "logs" / args.env /
                 (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}'),
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
