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
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune
from ray.tune import Stopper
from ray.tune import ResultGrid


path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
print(path2)
# os.path.join()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing"
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
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
                num_seconds=18000,  # 仿真时长
                reward_fn="calc_bottlespeed",
                delta_time=60,  # 控制时长
            )
        ),


    )

    config = PPOConfig()
    config.environment(env="vsl_merging")
    config.rollouts(batch_mode='complete_episodes')




    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(),
        param_space={
            "env": "vsl_merging",
            "num_workers": 8,
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "sgd_minibatch_size": tune.choice([128, 512, 2048]),
            "train_batch_size": tune.choice([10000, 20000, 40000]),
        },
        run_config=air.RunConfig(stop={"training_iteration": 100},
                                 local_dir="outputs/ray_results"
                                 ),
    )
    results = tuner.fit()

