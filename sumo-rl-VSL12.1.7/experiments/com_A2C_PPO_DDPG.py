
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
from ray.tune import Stopper
from ray.tune import ResultGrid
import time
path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
print(path2)

#"calc_jam"
REWARD = "calc_bottlespeed"
training_iteration = 300

def A2C_vsl():
    ray.init()
    register_env(
        "vsl_merging",
        lambda _: PettingZooEnv(
            sumo_rl.env(
                net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
                route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
                out_csv_name="outputs/ramp_vsl",
                use_gui=False,
                write_newtrips=False,
                num_seconds=7200,  # 仿真时长
                 delta_time=60,  # 控制时长
                reward_fn=REWARD
            )
        ),
    )

    config = A2CConfig()
    config.environment(env="vsl_merging")
    config.rollouts(batch_mode='complete_episodes')   # num_rollout_workers=14, create_env_on_local_worker=True,
    config.training(lr=tune.grid_search([0.01, 0.001, 0.0001]))



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




def PPO_vsl():
    ray.shutdown()
    ray.init()
    register_env(
        "vsl_merging",
        lambda _: PettingZooEnv(
            sumo_rl.env(
                net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
                route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
                out_csv_name="outputs/ramp_vsl",
                use_gui=False,
                write_newtrips=False,
                num_seconds=7200,  # 仿真时长
                 delta_time=60,  # 控制时长
                reward_fn=REWARD
            )
        ),
    )

    config = PPOConfig()
    config.environment(env="vsl_merging")
    config.rollouts(batch_mode='complete_episodes')
    config.training(lr=tune.grid_search([0.01, 0.001, 0.0001]))

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

def DDPG_vsl():
    ray.shutdown()
    ray.init()
    register_env(
        "vsl_merging",
        lambda _: PettingZooEnv(
            sumo_rl.env(
                net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
                route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
                out_csv_name="outputs/ramp_vsl",
                use_gui=False,
                write_newtrips=False,
                num_seconds=5400,  # 仿真时长
                 delta_time=60,  # 控制时长
                reward_fn=REWARD
            )
        ),


    )

    config = DDPGConfig()
    config.environment(env="vsl_merging")
    config.rollouts(batch_mode='complete_episodes')   # num_rollout_workers=14, create_env_on_local_worker=True,
    config.training(lr=tune.grid_search([0.01, 0.001, 0.0001]))
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

    # A2C_vsl()
    # time.sleep(10)
    # PPO_vsl()
    # time.sleep(10)
    DDPG_vsl()