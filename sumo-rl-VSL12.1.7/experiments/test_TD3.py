
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
from ray.rllib.algorithms.td3 import TD3Config
from ray import air
from ray import tune
from ray.tune import Stopper
from ray.tune import ResultGrid


path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
print(path2)
# os.path.join()


if __name__ == "__main__":
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
                num_seconds=5400,  # 仿真时长
                delta_time=30,  # 控制时长
                reward_fn="calc_bottlespeed"
            )
        ),
    )

    config = TD3Config()
    config.environment(env="vsl_merging")
    config.rollouts(batch_mode='complete_episodes')



    tuner = tune.Tuner(
        "TD3",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop={"training_iteration": 30},
                                 local_dir="outputs/ray_results"
                                 ),

    )

    result_grid: ResultGrid = tuner.fit()

    num_results = len(result_grid)
    print("Number of results:", num_results)

