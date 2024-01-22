"""Evaluates the baseline performance of grid1 without RL control.

Baseline is an actuated traffic light provided by SUMO.
"""

import os
import sys
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import sumo_rl
path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
print(path2)

N_CPUS = 10

N_ROLLOUTS = N_CPUS * 4
HORIZON =20  #500

Training_iteration = 3



def grid1_baseline():
        env = sumo_rl.env(
                net_file=os.path.join(path2, "nets\\ramp_VSL\\map.net.xml"),
                route_file=os.path.join(path2, "nets\\ramp_VSL\\fcd.rou.xml"),
                det_file=os.path.join(path2, "nets\\ramp_VSL\\e1.add.xml"),
                out_csv_name="outputs/ramp_vsl",
                use_gui=False,
                write_newtrips=False,
                begin_time = 0,
                num_seconds=3000,  # 仿真时长
                delta_time=30,  # 控制时长
            )

        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if truncation == True:
                print("结束啦")
                env.close()
                break
            else:
                action = env.action_space(agent).sample()  # this is where you would insert your policy
                env.step(action)

if __name__ == '__main__':
    grid1_baseline()

