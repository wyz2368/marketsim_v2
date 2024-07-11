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


#MM
from marketsim.wrappers.MM_wrapper import MMEnv
from marketsim.MM.utils import write_to_csv
from utils import replace_inf_with_nearest_2d


from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env




def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--game_name", type=str, default="RLMM", help="Game name.")
    parser.add_argument("--root_result_folder", type=str, default='./root_result_RL',
                        help="Root directory of saved results")
    parser.add_argument("--num_iteration", type=int, default=1, help="Number of iterations")

    parser.add_argument("--num_background_agents", type=int, default=25, help="Number of background agents.")
    parser.add_argument("--sim_time", type=int, default=int(1e3), help="Simulation time.")
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

    #RL
    parser.add_argument("--training_num", type=int, default=1, help="Number of training environments")
    parser.add_argument("--test_num", type=int, default=1, help="Number of testing environments")
    parser.add_argument("--total_timesteps", type=int, default=int(1e4), help="total_timesteps")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to resume training")
    parser.add_argument("--subproc", type=bool, default=False, help="Use subprocessing")

    return parser.parse_args()


def make_env(args):
    normalizers = {"fundamental": 1e5, "reward":1e4, "min_order_val": 1e5, "invt": 10, "cash": 1e6}
    print("normalizers:", normalizers)

    def init_env():
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

    return init_env

def train_MM(args, checkpoint_dir):

    if args.subproc:
        train_envs = make_vec_env(make_env(args), n_envs=args.training_num, monitor_dir=checkpoint_dir, vec_env_cls=SubprocVecEnv)
        test_envs = make_vec_env(make_env(args), n_envs=args.test_num, monitor_dir=checkpoint_dir, vec_env_cls=SubprocVecEnv)
    else:
        train_envs = Monitor(make_env(args)(), checkpoint_dir)
        test_envs = Monitor(make_env(args)(), checkpoint_dir)

    # eval_callback = EvalCallback(test_envs, best_model_save_path=checkpoint_dir,
    #                              log_path=checkpoint_dir, eval_freq=max(500 // args.training_num, 1),
    #                              n_eval_episodes=5, deterministic=True,
    #                              render=False)

    model = SAC(policy="MlpPolicy",
                env=train_envs,
                verbose=2,
                tensorboard_log=checkpoint_dir)

    # model.learn(total_timesteps=args.total_timesteps,
    #             callback=eval_callback,
    #             progress_bar=True)

    # model.learn(total_timesteps=args.total_timesteps,
    #             progress_bar=True)

    return model


def evaluation(model, args, checkpoint_dir):
    all_spreads, all_midprices, all_inventory, all_tq, all_MM_q, MM_values = [], [], [], [], [], []

    env = make_env(args)()

    rewards = []

    for i in range(args.num_iteration):
        print("Current Iter:", i)
        obs, info = env.reset()
        while env.time < args.sim_time:
            action, _states = model.predict(obs, deterministic=True)
            # print("ACT:", action)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

        print("-------------")
        print("REW:", rewards)
        print("RET:", np.sum(rewards))


        stats = env.get_stats()
        all_spreads.append(stats["spreads"])
        all_midprices.append(stats["midprices"])
        all_inventory.append(stats["inventory"])
        all_tq.append(stats["total_quantity"])
        all_MM_q.append(stats["MM_quantity"])
        MM_values.append(stats["MM_value"])
        print("MM values:", stats["MM_value"])

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
    model = train_MM(checkpoint_dir=checkpoint_dir, args=args)
    print("=============== End of Training ================")

    print("=============== START of Evaluation ================")
    evaluation(model=model, args=args, checkpoint_dir=checkpoint_dir)
    print("=============== End of Evaluation ================")


if __name__ == "__main__":
    main()
