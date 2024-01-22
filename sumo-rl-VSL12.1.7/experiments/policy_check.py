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
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False

from ray.rllib.algorithms.ppo import PPOConfig
path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
print(path2)

REWARD = "calc_bottlespeed"

path3 ="experiments\\outputs\\ray_results\\A2C\\A2C_vsl_merging_5c47f_00000_0_2023-07-27_11-35-52\\checkpoint_000400"

p_dir = os.path.join(path2,path3)
#
if __name__ == "__main__":
    ray.init()
    env = sumo_rl.env(
                net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
                route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
                det_file=os.path.join(path2, "nets\\ramp_VSL\\e1.add.xml"),
                out_csv_name="outputs/ramp_vsl",
                use_gui=False,
                write_newtrips=False,
                num_seconds=5400,  # 仿真时长
                 delta_time=30,  # 控制时长
                reward_fn=REWARD
            )


    algo = Algorithm.from_checkpoint(p_dir)

    episode_reward = 0
    terminated = truncated = False

    if gymnasium:
        obs, info = env.reset()
    else:
        obs = env.reset()

    while not terminated and not truncated:
        action = algo.compute_single_action(obs)
        if gymnasium:
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            obs, reward, terminated, info = env.step(action)
        episode_reward += reward


