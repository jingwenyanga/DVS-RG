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
from ray import air
from ray import tune
from ray.tune import Stopper
from ray.tune import ResultGrid


path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
print(path2)
# os.path.join()

REWARD = "calc_bottlespeed"

if __name__ == "__main__":
    ray.init()
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
                begin_time = 0,
                num_seconds = 5400,  # 仿真时长
                 delta_time=30,  # 控制时长
                reward_fn=REWARD
            )
        ),
    )

    config = A2CConfig()
    config.environment(env="vsl_merging")
    config.rollouts(batch_mode='complete_episodes')   # num_rollout_workers=14, create_env_on_local_worker=True,



    tuner = tune.Tuner(
        "A2C",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop={"training_iteration": 1},
                                 local_dir="outputs/ray_results"
                                 ),

    )

    result_grid: ResultGrid = tuner.fit()

    num_results = len(result_grid)
    print("Number of results:", num_results)

