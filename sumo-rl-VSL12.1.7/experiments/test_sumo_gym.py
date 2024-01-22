import os
import sys


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
from gym import spaces
# from ray.rllib.agents.a3c.a3c import A3CTrainer
#
# # from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
import gymnasium as gym
import sumo_rl
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.a3c import A3C
from ray.rllib.algorithms.ppo import PPOConfig

if __name__ == "__main__":
    ray.init()
    register_env(
        "4x4grid",
        lambda _: PettingZooEnv(
            sumo_rl.env(
                net_file="../nets/ramp_VSL/map.net.xml",
                route_file="../nets/ramp_VSL/fcd.rou.xml",
                det_file="../nets/ramp_VSL/ns.det.xml",
                out_csv_name="outputs/ramp_vsl",
                use_gui=True,
                write_newtrips = True,
                num_seconds=8000,
            )
        ),
    )
    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment("4x4grid")
        .rollouts(num_rollout_workers=1)
        .evaluation(evaluation_num_workers=1)
    )

    algo = config.build()  # 2. build the algorithm,


    algo.train()  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.


    #
    # algo = A3C(env="4x4grid", config= config)
    #
    # while True:
    #     print(algo.train())

