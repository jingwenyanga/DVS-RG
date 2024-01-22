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
from ray.rllib.agents.a3c.a3c import A3CTrainer
import sumo_rl
from ray.rllib.algorithms import ppo

if __name__ == "__main__":
    ray.init()

    register_env(
        "4x4grid",
        lambda _: PettingZooEnv(
            sumo_rl.env(
                net_file="../nets/ramp",
                route_file="../nets/single-intersection/single-intersection.rou.xml",
                out_csv_name="outputs/4x4grid/a3c",
                use_gui=False,
                write_newtrips=False,
                num_seconds=80000,
            )
        ),
    )
    env = "4x4grid"
    print(env.action_space())


    #
    # algo = A3CTrainer(env="4x4grid")
    #
    # while True:
    #     print(algo.train())

