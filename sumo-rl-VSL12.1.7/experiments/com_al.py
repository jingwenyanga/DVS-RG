

import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import ray
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
import sumo_rl
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune
import time
from ray.tune import ResultGrid
path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
print(path2)

#"calc_jam"
#REWARD = "bott_speed_rjam"  #calc_bottlespeed  cal_in_outflow calc_com_speed_flow
training_iteration = 1
Num_cpus = 16
num_seconds = 5400  #3600-5400--> 1800
delta_time = 30 #

def A2C_vsl(REWARD):
    ray.shutdown()
    ray.init(
      num_cpus=Num_cpus,
      include_dashboard=False,
      ignore_reinit_error=True,
      log_to_driver=False,
    )

    """环境注册部分"""
    register_env(
        "vsl_merging",
        lambda _: PettingZooEnv(
            sumo_rl.env(
                net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
                route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
                det_file=os.path.join(path2, "nets\\ramp_VSL\\e1.add.xml"),
                out_csv_name="outputs/ramp_vsl",
                use_gui=False,
                write_newtrips=False,
                num_seconds=num_seconds,  # 仿真时长
                 delta_time=delta_time,  # 控制时长
                reward_fn=REWARD
            )
        ),
    )

    """A2C"""


    config = A2CConfig()
    config.environment(env="vsl_merging")
    config.training(lr=tune.grid_search([0.01, 0.001, 0.0001]))
    config.rollouts(num_rollout_workers=7)  # num_rollout_workers=14, create_env_on_local_worker=True,
    config.resources(num_cpus_per_worker=2)
    config.train_batch_size = 128

    tuner = tune.Tuner(
        "A2C",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop={"training_iteration": training_iteration},
                                 local_dir="outputs/ray_results"
                                 ),

    )

    result_grid: ResultGrid = tuner.fit()

    num_results = len(result_grid)
    print("Number of results:", num_results)


def PPO_vsl(REWARD,train_batch_size):
    ray.shutdown()
    """环境注册部分"""
    register_env(
        "vsl_merging",
        lambda _: PettingZooEnv(
            sumo_rl.env(
                net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
                route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
                det_file=os.path.join(path2, "nets\\ramp_VSL\\e1.add.xml"),
                out_csv_name="outputs/ramp_vsl",
                use_gui=False,
                write_newtrips=False,
                num_seconds=num_seconds,  # 仿真时长
                delta_time=delta_time,  # 控制时长
                reward_fn=REWARD
            )
        ),
    )




    """PPO"""

    ray.init(
      num_cpus=Num_cpus,
      include_dashboard=False,
      ignore_reinit_error=True,
      log_to_driver=False,
    )

    config = PPOConfig()
    config.environment(env="vsl_merging")
    config.rollouts(num_rollout_workers=5)   # num_rollout_workers=14, create_env_on_local_worker=True,
    config.resources(num_cpus_per_worker=2)
    config.training(lr=tune.grid_search([0.01, 0.001, 0.0001]))
    config.train_batch_size = train_batch_size

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop={"training_iteration": training_iteration},
                                 local_dir="outputs/ray_results"
                                 ),

    )

    result_grid: ResultGrid = tuner.fit()
    num_results = len(result_grid)
    print("Number of results:", num_results)


def DDPG_vsl(REWARD1):
    """环境注册部分"""
    ray.shutdown()
    register_env(
        "vsl_merging",
        lambda _: PettingZooEnv(
            sumo_rl.env(
                net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
                route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
                det_file=os.path.join(path2, "nets\\ramp_VSL\\e1.add.xml"),
                out_csv_name="outputs/ramp_vsl",
                use_gui=False,
                write_newtrips=False,
                num_seconds=num_seconds,  # 仿真时长
                delta_time=delta_time,  # 控制时长
                reward_fn=REWARD1
            )
        ),
    )

    """ddpg"""
    ray.init(
      num_cpus=Num_cpus,
      include_dashboard=False,
      ignore_reinit_error=True,
      log_to_driver=False,
    )

    config = DDPGConfig()
    config.environment(env="vsl_merging")
    # config.rollouts(batch_mode='complete_episodes')   # num_rollout_workers=14, create_env_on_local_worker=True,
    config.training(lr=tune.grid_search([0.01, 0.001, 0.0001]))
    config.rollouts(num_rollout_workers=7)
                    # batch_mode='complete_episodes')  # num_rollout_workers=14, create_env_on_local_worker=True,
    config.resources(num_cpus_per_worker=2)
    # config.train_batch_size = train_batch_size

    tuner = tune.Tuner(
        "DDPG",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop={"training_iteration": training_iteration},
                                 local_dir="outputs/ray_results"
                                 ),

    )

    result_grid: ResultGrid = tuner.fit()
    num_results = len(result_grid)
    print("Number of results:", num_results)

if __name__ == "__main__":

    A2C_vsl("calc_bottlespeed")

    for j in [256, 512, 1024, 2048]:
        PPO_vsl("calc_bottlespeed", j)


    DDPG_vsl("calc_bottlespeed")
