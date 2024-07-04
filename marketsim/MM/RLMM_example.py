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
from absl import app
from absl import flags

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from marketsim.wrappers.metrics import sharpe_ratio

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
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

FLAGS = flags.FLAGS

# Sim setup

flags.DEFINE_string("game_name", "RLMM", "Game name.")
flags.DEFINE_string("root_result_folder", './root_result_RL', "root directory of saved results")
flags.DEFINE_integer("num_iteration", 2, "num_iteration")

flags.DEFINE_integer("num_background_agents", 25, "Number of background agents.")
flags.DEFINE_integer("sim_time", int(1e4), "Simulation time.")
flags.DEFINE_float("lam", 0.075, "Lambda.")
flags.DEFINE_float("lamMM", 0.005, "Lambda MM.")
flags.DEFINE_float("mean", 1e5, "Mean.")
flags.DEFINE_float("r", 0.05, "Interest rate.")
flags.DEFINE_float("shock_var", 5e6, "Shock variance.")
flags.DEFINE_integer("q_max", 10, "Maximum quantity.")
flags.DEFINE_float("pv_var", 5e6, "PV variance.")
flags.DEFINE_list("shade", [250, 500], "Shade.")
flags.DEFINE_integer("xi", 50, "Rung size.")
flags.DEFINE_integer("omega", 10, "Spread.")
flags.DEFINE_integer("K", 20, "Number of levels - 1.")
flags.DEFINE_integer("n_levels", 21, "n_levels.")
flags.DEFINE_integer("total_volume", 100, "total_volume.")
flags.DEFINE_boolean("policy", True, "Policy.")
flags.DEFINE_boolean("beta_MM", True, "Beta MM.")
flags.DEFINE_boolean("inv_driven", False, "Inventory driven.")
flags.DEFINE_integer("w0", 5, "Initial wealth.")
flags.DEFINE_integer("p", 2, "Parameter p.")
flags.DEFINE_integer("k_min", 5, "Minimum k.")
flags.DEFINE_integer("k_max", 20, "Maximum k.")
flags.DEFINE_integer("max_position", 20, "Maximum position.")
flags.DEFINE_boolean("agents_only", False, "agents_only.")


# RL setup
flags.DEFINE_string("task", "RLMM", "Task name")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("buffer_size", 1000000, "Buffer size")
flags.DEFINE_float("actor_lr", 3e-4, "Learning rate for actor")
flags.DEFINE_float("critic_lr", 1e-3, "Learning rate for critic")
flags.DEFINE_float("gamma", 0.99, "Discount factor")
flags.DEFINE_float("tau", 0.005, "Target smoothing coefficient")
flags.DEFINE_float("alpha", 0.1, "Entropy regularization coefficient")
flags.DEFINE_integer("auto_alpha", 1, "Automatic tuning of alpha")
flags.DEFINE_float("alpha_lr", 3e-4, "Learning rate for alpha")
flags.DEFINE_integer("epoch", 2, "Number of epochs")
flags.DEFINE_integer("step_per_epoch", 100, "Number of steps per epoch")
flags.DEFINE_integer("step_per_collect", 10, "Number of steps per collect")
flags.DEFINE_float("update_per_step", 0.1, "Update per step")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_list("hidden_sizes", [128, 128], "Hidden sizes of the network")
flags.DEFINE_integer("training_num", 1, "Number of training environments")
flags.DEFINE_integer("test_num", 1, "Number of testing environments")
flags.DEFINE_string("logdir", "root_results", "Directory to save logs")
flags.DEFINE_float("render", 0.0, "Render frequency")
flags.DEFINE_integer("n_step", 4, "N-step return")
flags.DEFINE_string("device", "cuda" if torch.cuda.is_available() else "cpu", "Device to use")
flags.DEFINE_string("resume_path", None, "Path to resume training")
flags.DEFINE_boolean("subproc", False, "Use subprocessing")


def make_env():
    normalizers = {"fundamental": 1e5, "reward":1e2, "min_order_val": 1e5, "invt": 10, "cash": 1e6}
    print("normalizers:", normalizers)

    env = MMEnv(num_background_agents = FLAGS.num_background_agents,
                sim_time = FLAGS.sim_time,
                lam = FLAGS.lam,
                lamMM = FLAGS.lamMM,
                mean = FLAGS.mean,
                r = FLAGS.r,
                shock_var = FLAGS.shock_var,
                q_max = FLAGS.q_max,
                pv_var = FLAGS.pv_var,
                shade = FLAGS.shade,
                xi = FLAGS.xi,
                omega = FLAGS.omega,
                n_levels=FLAGS.n_levels,
                total_volume=FLAGS.total_volume,
                policy = FLAGS.policy,
                normalizers=normalizers)

    return env

def train_MM(checkpoint_dir):
    env = make_env()
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    if FLAGS.subproc:
        train_envs = SubprocVectorEnv(
            [lambda: make_env() for _ in range(FLAGS.training_num)],
        )
        # test_envs = gym.make(FLAGS.task)
        test_envs = SubprocVectorEnv(
            [
                lambda: make_env() for _ in range(FLAGS.test_num)
            ],
        )
    else:
        train_envs = DummyVectorEnv(
            [lambda: make_env() for _ in range(FLAGS.training_num)],
        )
        test_envs = DummyVectorEnv(
            [
                lambda: make_env() for _ in range(FLAGS.test_num)
            ],
        )

    # seed
    # np.random.seed(FLAGS.seed)
    # torch.manual_seed(FLAGS.seed)
    # train_envs.seed(FLAGS.seed)
    # test_envs.seed(FLAGS.seed)

    # model
    net_a = Net(state_shape=state_shape, hidden_sizes=FLAGS.hidden_sizes, device=FLAGS.device)
    actor = ActorProb(
        preprocess_net=net_a,
        action_shape=action_shape,
        device=FLAGS.device,
        unbounded=True,
    ).to(FLAGS.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=FLAGS.actor_lr)

    net_c1 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=FLAGS.hidden_sizes,
        concat=True,
        device=FLAGS.device,
    )
    critic1 = Critic(net_c1, device=FLAGS.device).to(FLAGS.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=FLAGS.critic_lr)

    net_c2 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=FLAGS.hidden_sizes,
        concat=True,
        device=FLAGS.device,
    )
    critic2 = Critic(net_c2, device=FLAGS.device).to(FLAGS.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=FLAGS.critic_lr)

    action_dim = space_info.action_info.action_dim
    if FLAGS.auto_alpha:
        target_entropy = -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=FLAGS.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=FLAGS.alpha_lr)
        FLAGS.alpha = (target_entropy, log_alpha, alpha_optim)

    policy: SACPolicy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=FLAGS.tau,
        gamma=FLAGS.gamma,
        alpha=FLAGS.alpha,
        estimation_step=FLAGS.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if FLAGS.resume_path:
        policy.load_state_dict(torch.load(FLAGS.resume_path))
        print("Loaded agent from: ", FLAGS.resume_path)

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(FLAGS.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=FLAGS.buffer_size)

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
        max_epoch=FLAGS.epoch,
        step_per_epoch=FLAGS.step_per_epoch,
        step_per_collect=FLAGS.step_per_collect,
        episode_per_test=FLAGS.test_num,
        batch_size=FLAGS.batch_size,
        update_per_step=FLAGS.update_per_step,
        test_in_train=False,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    print("Pretty print of the result given by OffpolicyTrainer.")
    pprint.pprint(result)

    # Evaluation.
    # policy.eval()
    # test_envs.seed(FLAGS.seed)
    # test_collector.reset()
    # collector_stats = test_collector.collect(n_episode=FLAGS.test_num, render=FLAGS.render)
    # print(collector_stats)

    return policy


def evaluation(policy, checkpoint_dir):
    policy.eval()
    all_spreads, all_midprices, all_inventory, all_tq, all_MM_q, MM_values = [], [], [], [], [], []

    env = make_env()
    for i in range(FLAGS.num_iteration):
        print("Current Iter:", i)
        obs, info = env.reset()
        while env.time < FLAGS.sim_time:
            action = policy.predict(obs)  # this is where you would insert your policy
            obs, reward, terminated, truncated, info = env.step(action)

        stats = env.get_stats()
        all_spreads.append(stats["spreads"])
        all_midprices.append(stats["midprices"])
        all_inventory.append(stats["inventory"])
        all_tq.append(stats["total_quantity"])
        all_MM_q.append(stats["MM_quantity"])
        MM_values.append(stats["MM_value"])

    # Remove inf
    all_spreads = replace_inf_with_nearest_2d(all_spreads)
    all_midprices = replace_inf_with_nearest_2d(all_midprices)

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


def main(argv):
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)

    seed = np.random.randint(0, 10000)

    checkpoint_dir = FLAGS.game_name
    checkpoint_dir = checkpoint_dir + "_RL" + "_se_" + str(
        seed) + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save the original standard output
    # sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')

    print("========== Parameters ==========")
    print(f"game_name: {FLAGS.game_name}")
    print(f"root_result_folder: {FLAGS.root_result_folder}")
    print(f"num_iteration: {FLAGS.num_iteration}")
    print(f"num_background_agents: {FLAGS.num_background_agents}")
    print(f"sim_time: {FLAGS.sim_time}")
    print(f"lam: {FLAGS.lam}")
    print(f"lamMM: {FLAGS.lamMM}")
    print(f"mean: {FLAGS.mean}")
    print(f"r: {FLAGS.r}")
    print(f"shock_var: {FLAGS.shock_var}")
    print(f"q_max: {FLAGS.q_max}")
    print(f"pv_var: {FLAGS.pv_var}")
    print(f"shade: {FLAGS.shade}")
    print(f"xi: {FLAGS.xi}")
    print(f"omega: {FLAGS.omega}")
    print(f"n_levels: {FLAGS.n_levels}")
    print(f"K: {FLAGS.K}")
    print(f"total_volume: {FLAGS.total_volume}")
    print(f"policy: {FLAGS.policy}")
    print(f"beta_MM: {FLAGS.beta_MM}")
    print(f"inv_driven: {FLAGS.inv_driven}")
    print(f"w0: {FLAGS.w0}")
    print(f"p: {FLAGS.p}")
    print(f"k_min: {FLAGS.k_min}")
    print(f"k_max: {FLAGS.k_max}")
    print(f"max_position: {FLAGS.max_position}")

    print("=============== START of Training ================")
    policy = train_MM(checkpoint_dir=checkpoint_dir)
    print("=============== End of Training ================")

    # print("=============== START of Evaluation ================")
    # evaluation(policy=policy, checkpoint_dir=checkpoint_dir)
    # print("=============== End of Evaluation ================")


if __name__ == "__main__":
    app.run(main)
