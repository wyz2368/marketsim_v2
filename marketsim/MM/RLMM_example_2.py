import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Get the grandparent directory
grandparent_dir = os.path.dirname(parent_dir)

# Add the grandparent directory to the sys.path
sys.path.append(grandparent_dir)

import pprint
import functools
print = functools.partial(print, flush=True)
import datetime
from typing import cast
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, Batch, to_numpy
from tianshou.data.types import ObsBatchProtocol
from tianshou.env import SubprocVectorEnv, DummyVectorEnv, RayVectorEnv
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.space_info import SpaceInfo

#MM
from marketsim.wrappers.MM_wrapper import MMEnv
from marketsim.MM.utils import write_to_csv
from utils import replace_inf_with_nearest_2d


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--game_name", type=str, default="RLMM", help="Game name.")
    parser.add_argument("--root_result_folder", type=str, default='./root_result_RL',
                        help="Root directory of saved results")
    parser.add_argument("--num_iteration", type=int, default=200, help="Number of iterations")

    parser.add_argument("--num_background_agents", type=int, default=25, help="Number of background agents.")
    parser.add_argument("--sim_time", type=int, default=int(1e5), help="Simulation time.")
    parser.add_argument("--lam", type=float, default=0.075, help="Lambda.")
    parser.add_argument("--lamMM", type=float, default=0.005, help="Lambda MM.")
    parser.add_argument("--mean", type=float, default=1e5, help="Mean.")
    parser.add_argument("--r", type=float, default=0.05, help="Interest rate.")
    parser.add_argument("--shock_var", type=float, default=5e6, help="Shock variance.")
    parser.add_argument("--q_max", type=int, default=10, help="Maximum quantity.")
    parser.add_argument("--pv_var", type=float, default=5e6, help="PV variance.")
    parser.add_argument("--shade", type=int, nargs='+', default=[250, 500], help="Shade.")
    parser.add_argument("--xi", type=int, default=50, help="Rung size.")
    parser.add_argument("--omega", type=int, default=10, help="Spread.")
    parser.add_argument("--K", type=int, default=20, help="Number of levels - 1.")
    parser.add_argument("--n_levels", type=int, default=21, help="Number of levels.")
    parser.add_argument("--total_volume", type=int, default=100, help="Total volume.")
    parser.add_argument("--policy", type=bool, default=True, help="Policy.")
    parser.add_argument("--beta_MM", type=bool, default=True, help="Beta MM.")
    parser.add_argument("--inv_driven", type=bool, default=False, help="Inventory driven.")
    parser.add_argument("--w0", type=int, default=5, help="Initial wealth.")
    parser.add_argument("--p", type=int, default=2, help="Parameter p.")
    parser.add_argument("--k_min", type=int, default=5, help="Minimum k.")
    parser.add_argument("--k_max", type=int, default=20, help="Maximum k.")
    parser.add_argument("--max_position", type=int, default=20, help="Maximum position.")
    parser.add_argument("--agents_only", type=bool, default=False, help="Agents only.")

    parser.add_argument("--task", type=str, default="RLMM", help="Task name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Buffer size")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Learning rate for actor")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Learning rate for critic")
    parser.add_argument("--gamma", type=float, default=1.00, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target smoothing coefficient")
    parser.add_argument("--alpha", type=float, default=0.1, help="Entropy regularization coefficient")
    parser.add_argument("--auto_alpha", type=int, default=1, help="Automatic tuning of alpha")
    parser.add_argument("--alpha_lr", type=float, default=3e-4, help="Learning rate for alpha")
    parser.add_argument("--epoch", type=int, default=30, help="Number of epochs")
    parser.add_argument("--step_per_epoch", type=int, default=50000, help="Number of steps per epoch")
    parser.add_argument("--step_per_collect", type=int, default=10, help="Number of steps per collect")
    parser.add_argument("--update_per_step", type=float, default=0.1, help="Update per step")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[256, 256], help="Hidden sizes of the network")
    parser.add_argument("--training_num", type=int, default=4, help="Number of training environments")
    parser.add_argument("--test_num", type=int, default=4, help="Number of testing environments")
    parser.add_argument("--episode_per_test", type=int, default=5, help="Number of episodes per test")
    parser.add_argument("--logdir", type=str, default="root_results", help="Directory to save logs")
    parser.add_argument("--render", type=float, default=0.0, help="Render frequency")
    parser.add_argument("--n_step", type=int, default=4, help="N-step return")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to resume training")
    parser.add_argument("--subproc", type=bool, default=True, help="Use subprocessing")


    return parser.parse_args()


def make_env(args):
    normalizers = {"fundamental": 1e5, "reward":1e2, "min_order_val": 1e5, "invt": 10, "cash": 1e6}
    print("normalizers:", normalizers)

    env = MMEnv(num_background_agents = args.num_background_agents,
                sim_time = args.sim_time,
                lam = args.lam,
                lamMM = args.lamMM,
                mean = args.mean,
                r = args.r,
                shock_var = args.shock_var,
                q_max = args.q_max,
                pv_var = args.pv_var,
                shade = args.shade,
                xi = args.xi,
                omega = args.omega,
                n_levels=args.n_levels,
                total_volume=args.total_volume,
                policy = args.policy,
                normalizers=normalizers)

    return env

def train_MM(checkpoint_dir, args):
    env = make_env(args)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    if args.subproc:
        train_envs = SubprocVectorEnv(
            [lambda: make_env(args) for _ in range(args.training_num)],
        )
        # test_envs = gym.make(args.task)
        test_envs = SubprocVectorEnv(
            [
                lambda: make_env(args) for _ in range(args.test_num)
            ],
        )
    else:
        train_envs = DummyVectorEnv(
            [lambda: make_env(args) for _ in range(args.training_num)],
        )
        test_envs = DummyVectorEnv(
            [
                lambda: make_env(args) for _ in range(args.test_num)
            ],
        )

    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # model
    net_a = Net(state_shape=state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        preprocess_net=net_a,
        action_shape=action_shape,
        device=args.device,
        unbounded=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    net_c2 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    action_dim = space_info.action_info.action_dim
    if args.auto_alpha:
        target_entropy = -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy: SACPolicy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
        action_scaling=True
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.buffer_size)

    writer = SummaryWriter(checkpoint_dir)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(checkpoint_dir, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        test_in_train=False,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    print("Pretty print of the result given by OffpolicyTrainer.")
    pprint.pprint(result)

    # Evaluation.
    print("=============== START of TEST ================")
    policy.eval()
    # test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.num_iteration, render=args.render)
    print(collector_stats)

    return policy


def evaluation(policy, args, checkpoint_dir):
    policy.eval()
    all_spreads, all_midprices, all_inventory, all_tq, all_MM_q, MM_values = [], [], [], [], [], []

    env = make_env(args)
    for i in range(args.num_iteration):
        print("Current Iter:", i)
        obs, info = env.reset()
        while env.time < args.sim_time:
            obs_batch = cast(ObsBatchProtocol, Batch(obs=np.expand_dims(obs, axis=0), info=Batch()))
            act_batch = policy(obs_batch)  # this is where you would insert your policy
            # print("ACT:", to_numpy(act_batch.act))
            action = to_numpy(act_batch.act)[0]
            action = np.clip(action, 0.01, 1)
            obs, reward, terminated, truncated, info = env.step(action)


        stats = env.get_stats()
        all_spreads.append(stats["spreads"])
        all_midprices.append(stats["midprices"])
        all_inventory.append(stats["inventory"])
        all_tq.append(stats["total_quantity"])
        all_MM_q.append(stats["MM_quantity"])
        MM_values.append(stats["MM_value"])
        # print("STATS:", stats["midprices"])

    # Remove inf
    all_spreads = replace_inf_with_nearest_2d(all_spreads)
    all_midprices = replace_inf_with_nearest_2d(all_midprices)

    # print("STATS:", all_midprices)

    # Simulation Output
    average_spreads = np.mean(all_spreads, axis=0)
    average_midprices = np.mean(all_midprices, axis=0)
    average_inventory = np.mean(all_inventory, axis=0)
    average_tq = np.mean(all_tq)
    average_MM_q = np.mean(all_MM_q)
    average_values = np.mean(MM_values)

    average_spreads_std = np.std(all_spreads, axis=0)
    average_midprices_std = np.std(all_midprices, axis=0)
    average_inventory_std = np.std(all_inventory, axis=0)
    average_tq_std = np.std(all_tq)
    average_MM_q_std = np.std(all_MM_q)
    average_values_std = np.std(MM_values)

    print("Average Spreads:", np.mean(average_spreads))
    print("Average Midprices:", np.mean(average_midprices))
    print("Average Inventory:", np.mean(average_inventory))

    print("Average Total Quantity:", average_tq)
    print("Average MM Quantity:", average_MM_q)
    print("Average Values:", average_values)

    print("Std Total Quantity:", average_tq_std)
    print("Std MM Quantity:", average_MM_q_std)
    print("Std Values:", average_values_std)

    print("Social Welfare:", env.compute_social_welfare())

    # Save everything
    write_to_csv(checkpoint_dir + "/average_spreads.csv", average_spreads)
    write_to_csv(checkpoint_dir + "/average_midprices.csv", average_midprices)
    write_to_csv(checkpoint_dir + "/average_inventory.csv", average_inventory)

    write_to_csv(checkpoint_dir + "/average_spreads_std.csv", average_spreads_std)
    write_to_csv(checkpoint_dir + "/average_midprices_std.csv", average_midprices_std)
    write_to_csv(checkpoint_dir + "/average_inventory_std.csv", average_inventory_std)


def main():
    args = get_args()

    if not os.path.exists(args.root_result_folder):
        os.makedirs(args.root_result_folder)

    seed = np.random.randint(0, 10000)

    checkpoint_dir = args.game_name
    checkpoint_dir = checkpoint_dir + "_RL" + "_se_" + str(
        seed) + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), args.root_result_folder, checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save the original standard output
    # sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')

    print("========== Parameters ==========")
    print(f"game_name: {args.game_name}")
    print(f"root_result_folder: {args.root_result_folder}")
    print(f"num_iteration: {args.num_iteration}")
    print(f"num_background_agents: {args.num_background_agents}")
    print(f"sim_time: {args.sim_time}")
    print(f"lam: {args.lam}")
    print(f"lamMM: {args.lamMM}")
    print(f"mean: {args.mean}")
    print(f"r: {args.r}")
    print(f"shock_var: {args.shock_var}")
    print(f"q_max: {args.q_max}")
    print(f"pv_var: {args.pv_var}")
    print(f"shade: {args.shade}")
    print(f"xi: {args.xi}")
    print(f"omega: {args.omega}")
    print(f"n_levels: {args.n_levels}")
    print(f"K: {args.K}")
    print(f"total_volume: {args.total_volume}")
    print(f"policy: {args.policy}")
    print(f"beta_MM: {args.beta_MM}")
    print(f"inv_driven: {args.inv_driven}")
    print(f"w0: {args.w0}")
    print(f"p: {args.p}")
    print(f"k_min: {args.k_min}")
    print(f"k_max: {args.k_max}")
    print(f"max_position: {args.max_position}")

    print("=============== START of Training ================")
    policy = train_MM(checkpoint_dir=checkpoint_dir, args=args)
    print("=============== End of Training ================")

    # print("=============== START of Evaluation ================")
    # evaluation(policy=policy, args=args, checkpoint_dir=checkpoint_dir)
    # print("=============== End of Evaluation ================")


if __name__ == "__main__":
    main()
