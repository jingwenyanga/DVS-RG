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
import numpy as np
from sumo_rl.agents.ddpg_dvsl import VSL_DDPG_PR

path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
print(path2)
# os.path.join()
Control_excue = False
EP_MAX=10
if __name__ == "__main__":
    env = sumo_rl.env(
        net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
        route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
        det_file=os.path.join(path2, "nets\\ramp_VSL\\e1.add.xml"),
        out_csv_name="outputs/ramp_vsl",
        use_gui=False,
        write_newtrips=False,
        begin_time=3000,
        num_seconds=5400,  # 仿真时长
        delta_time=30,  # 控制时长
    )

    ddpg_agents = VSL_DDPG_PR(s_dim = 45, a_dim = 5)
    for ep in range(EP_MAX):
        env.reset()

        while True:
            observation, reward, termination, truncation, info = env.last()
            if truncation == True:
                print("结束啦")
                env.close()
                break
            else:
                action = ddpg_agents.choose_action(env.observe(env.agents[0]))  # this is where you would insert your policy
                env.step(action)
                ddpg_agents.store_transition(observation,action,env.rewards[env.agents[0]], env.observe(env.agents[0]))
                ddpg_agents.learn()

                # actions = {ts: ddpg_agents[ts].choose_action(env.observe(ts)) for ts in ddpg_agents.keys()}
                # print("+++++++++++++++", actions.values())
                # action2 = np.array(list(actions.values()))
                # env.step(action=np.squeeze(action2))
                # for agent_id in ddpg_agents.keys():
                #     ddpg_agents[agent_id].learn()

