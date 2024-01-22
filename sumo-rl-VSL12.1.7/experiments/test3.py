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
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray import air
from ray import tune
from ray.tune import Stopper
from ray.tune import ResultGrid


path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
print(path2)
# os.path.join()


if __name__ == "__main__":
    ray.init()
    register_env(
        "4x4grid",
        lambda _: PettingZooEnv(
            sumo_rl.env(
                net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
                route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
                out_csv_name="outputs/ramp_vsl",
                use_gui=False,
                write_newtrips=False,
                num_seconds=18000,  # 仿真时长
                delta_time=60,  # 控制时长
            )
        ),


    )

    config = DDPGConfig()
    config.environment(env="4x4grid")
    config.training(lr=tune.grid_search([0.01,0.001,0.0001]))
    config.rollouts(num_rollout_workers=3, batch_mode='complete_episodes')



    tuner = tune.Tuner(
        "DDPG",
        param_space=config.to_dict(),

        run_config=air.RunConfig(stop={"training_iteration": 10},
                                 local_dir="outputs/ray_results"
                                 ),

    )

    result_grid: ResultGrid = tuner.fit()

    num_results = len(result_grid)
    print("Number of results:", num_results)

