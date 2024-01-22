"""SUMO Environment for Traffic Signal Control."""
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci
from gymnasium.utils import EzPickle, seeding
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .observations import DefaultObservationFunction, ObservationFunction
# from .traffic_signal import TrafficSignal
from .traffic_node import TrafficNode
from collections import defaultdict
from sumo_rl.agents.ddpg_dvsl import VSL_DDPG_PR
"""路网参数"""
import os



M1FLOW = np.round(np.array([3359+640,6007+1229,5349+1080,5563+1139,5299+1107]))
R3FLOW = np.round(np.array([480,1153,1129,1176,1095]))
M1A = [0.75,0.25]
V_ratio = [0.1,0.1,0.4,0.4]

EDGES = ['m3 m4 m5 m6 m7 m8 m9',\
         'm3 m4 m5 m6 m7 m8 rout1',\
         'rlight1 rin3 m7 m8 m9']

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ
# list_value = ["12.5","13.89","15.27","16.67","18.05","19.44","20.83","22.22","23.61","27.77","29.16","30.55","31.94"]

list_value = ["18.05","19.44","20.83","22.22","23.61","27.77"]


Control_Value = float(list_value[0])
D_V = 0
D_IN = 1.38*3

def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZ(**kwargs)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.ClipOutOfBoundsWrapper(env)

    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)



# traci.start([sumolib.checkBinary("sumo"), "-n", "D:/project/python/ucd/RLlib_test/sumo-rl-VSL4/nets/ramp_VSL/map.net.xml"])

class SumoEnvironment(gym.Env):
    """SUMO Environment for Traffic Signal Control.

    Class that implements a gym.Env interface for traffic signal control using the SUMO simulator.
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://gymnasium.farama.org/ for details on gymnasium.

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        out_csv_name (Optional[str]): name of the .csv output with simulation results. If None, no output is generated
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (Optional[Tuple[int,int]]): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        num_seconds (int): Number of simulated seconds on SUMO. The time in seconds the simulation must end. Default: 3600
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): Simulation seconds between actions. Default: 5 seconds
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict): String with the name of the reward function used by the agents, a reward function, or dictionary with reward functions assigned to individual traffic lights by their keys.
        observation_class (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        fixed_ts (bool): If true, it will follow the phase configuration in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file: str,
        route_file: str,
        det_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        write_newtrips: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 3000,
        num_seconds: int = 20,   #仿真时长
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,   #控制时长
        single_agent: bool = False,
        reward_fn: Union[str, Callable, dict] = "cal_in_outflow",
        observation_class: ObservationFunction = DefaultObservationFunction,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        # additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:

        """Initialize the environment."""
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file
        self._route = route_file
        self._det = det_file

        self.use_gui = use_gui

        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        self.write_newtrips = write_newtrips
        self.begin_time = begin_time
        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport

        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        # self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        if LIBSUMO:
            traci.start([self._sumo_binary, "-n", self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            print(os.path.abspath(os.path.dirname(__file__)))
            print("---", self._sumo_binary)
            print("+++", self._net)
            traci.start([self._sumo_binary, "-n",self._net])
            conn = traci.getConnection()#"init_connection" + self.label

        self.nodes_ids = ["304822600"]
        self.observation_class = observation_class

        if isinstance(self.reward_fn, dict):
            self.traffic_nodes = {
                node: TrafficNode(
                    self,
                    node,
                    self.reward_fn[node],
                    conn,
                )
                for node in self.reward_fn.keys()
            }
        else:
            self.traffic_nodes = {
                node: TrafficNode(
                    self,
                    node,
                    self.reward_fn,
                    conn,
                )
                for node in self.nodes_ids
            }

        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {node: None for node in self.nodes_ids}
        self.rewards = {node: None for node in self.nodes_ids}
        self.reward_average = defaultdict(list)


    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self._route,
            "-a",
            self._det,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
            "--queue-output",
            "queue.xml",
        ]
        # if self._det is not None:
        #     sumo_cmd.extend(["-a", str(self._det)])

        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        # if self.sumo_seed == "random":
        #     sumo_cmd.append("--random")
        # else:
        #     sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")

        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")




        print(sumo_cmd, "++++++++++++++++")

        if LIBSUMO:
            if self.write_newtrips is not None:
                self.writenewtrip_s()
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            if self.write_newtrips == True:
                self.writenewtrip_s()

            traci.start(sumo_cmd)
            self.sumo = traci.getConnection()

        if self.use_gui or self.render_mode is not None:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")



    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)

        if self.episode != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        if isinstance(self.reward_fn, dict):
            self.traffic_nodes = {
                node: TrafficNode(
                    self,
                    node,
                    self.reward_fn[node],
                    self.sumo,
                )
                for node in self.reward_fn.keys()
            }
        else:
            self.traffic_nodes = {
                node: TrafficNode(
                    self,
                    node,
                    self.reward_fn,
                    self.sumo,
                )
                for node in self.nodes_ids
            }

        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.nodes_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()

    def _run_steps(self):
        self._sumo_step()

    def _apply_actions(self, actions):
        if self.single_agent:
            print("不是单智能体")
        else:
            for nodes, action in actions.items():
                vsl_main = ['m5_0', 'm5_1', 'm5_2', 'm5_3', 'm5_4']
                for i in range(len(vsl_main)):
                    action2 = self.from_a_to_mlv(action)
                    print(self.sim_step,"+++++++++++",vsl_main[i], action2[i])
                    self.sumo.lane.setMaxSpeed(vsl_main[i], action2[i])


    def _compute_dones(self):
        dones = {nodes_id: False for nodes_id in self.nodes_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info)
        return info

    def _compute_observations(self):
        self.observations.update(
            {node: self.traffic_nodes[node].compute_observation() for node in self.nodes_ids}
        )
        return {node: self.observations[node].copy() for node in self.observations.keys()}

    def _compute_rewards(self):
        self.rewards.update(
            {node: self.traffic_nodes[node].compute_reward() for node in self.nodes_ids if self.traffic_nodes[node]}
        )
        return {node: self.rewards[node] for node in self.rewards.keys() if self.traffic_nodes[node]}

    @property
    def observation_space(self):
        """Return the observation space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_nodes[self.nodes_ids[0]].observation_space

    @property
    def action_space(self):
        """Return the action space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_nodes[self.nodes_ids[0]].action_space

    def observation_spaces(self, nodes_id: str):
        """Return the observation space of a traffic signal."""
        return self.traffic_nodes[nodes_id].observation_space

    def action_spaces(self, nodes_id: str) -> gym.spaces.Box:
        """Return the action space of a traffic signal."""
        return self.traffic_nodes[nodes_id].action_space

    def _sumo_step(self):
        self.reward_average = defaultdict(list)
        for _ in range(self.delta_time):
            self.sumo.simulationStep()
            self.time_space_position_date()
            self.time_space_veh_date()
            for key,value in self._compute_rewards().items():
                self.reward_average[key].append(value)

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        emission_CO2 = [self.sumo.vehicle.getCO2Emission(vehicle) for vehicle in vehicles]
        fuelconsumption = [self.sumo.vehicle.getFuelConsumption(vehicle) for vehicle in vehicles]
        ttc_all = [self.sumo.vehicle.getParameter(vehicle, "device.ssm.minTTC") for vehicle in vehicles]

        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_get_in_flow": self.traffic_nodes[self.nodes_ids[0]].get_in_flow(),
            "system_total_get_out_flow": self.traffic_nodes[self.nodes_ids[0]].get_out_flow(),
            "system_get_merging_flow": self.traffic_nodes[self.nodes_ids[0]].get_merging_flow(),
            "system_total_VSL_lane_speed": self.traffic_nodes[self.nodes_ids[0]].get_VSL_area_speed(),
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            "system_emission_CO2": 0.0 if len(vehicles) == 0 else sum(emission_CO2) * self.delta_time,
            "system_fuelconsumption": 0.0 if len(vehicles) == 0 else sum(fuelconsumption)*self.delta_time,
            "system_total_TTC": sum(int(float(ttc) < 3.0) for ttc in ttc_all if len(ttc)!=0),
        }

    def _get_per_agent_info(self):
        stopped = [self.traffic_nodes[node].get_total_queued() for node in self.nodes_ids]
        accumulated_waiting_time = [
            sum(self.traffic_nodes[node].get_accumulated_waiting_time_per_lane()) for node in self.nodes_ids
        ]

        agent_state = [self.traffic_nodes[node]._observation_fn_default() for node in self.nodes_ids]
        average_speed = [self.traffic_nodes[ts].get_average_speed() for ts in self.nodes_ids]
        info = {}
        for i, node in enumerate(self.nodes_ids):
            info[f"{node}_stopped"] = stopped[i]
            info[f"{node}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{node}_average_speed"] = average_speed[i]
            info[f"{node}_agent_state"] = agent_state[i]

        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        if self.sumo is None:
            return
        #
        # if not LIBSUMO:
        #     traci.switch()
        traci.close()

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.sumo = None

    def __del__(self):
        """Close the environment and stop the SUMO simulation."""
        self.close()

    def render(self):
        """Render the environment.

        If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
        """
        if self.render_mode == "human":
            return  # sumo-gui will already be rendering the frame
        elif self.render_mode == "rgb_array":
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file.

        Args:
            out_csv_name (str): Path to the output .csv file. E.g.: "results/my_results
            episode (int): Episode number to be appended to the output file name.
        """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)

    def from_a_to_mlv(self, a):
        return 12.8816 + 2.2352 * np.floor(a)

    def time_space_position_date(self):
        with open(r'space_time_v_lane_r_position.csv', 'a') as f1:
            for veh1 in traci.lane.getLastStepVehicleIDs('rin3_0') +\
                        traci.lane.getLastStepVehicleIDs('m7_1'):

                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_0_positio.csv', 'a') as f1:
            for veh1 in traci.lane.getLastStepVehicleIDs('m4_0') +\
                        traci.lane.getLastStepVehicleIDs('m5_0') +\
                        traci.lane.getLastStepVehicleIDs('m6_0') +\
                        traci.lane.getLastStepVehicleIDs('m7_1'):

                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step/10))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_1_positio.csv', 'a') as f1:
            for veh1 in traci.lane.getLastStepVehicleIDs('m4_1') +\
                        traci.lane.getLastStepVehicleIDs('m5_1') +\
                        traci.lane.getLastStepVehicleIDs('m6_1') +\
                        traci.lane.getLastStepVehicleIDs('m7_2'):

                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_2_positio.csv', 'a') as f1:
            for veh1 in traci.lane.getLastStepVehicleIDs('m4_2') +\
                        traci.lane.getLastStepVehicleIDs('m5_2') +\
                        traci.lane.getLastStepVehicleIDs('m6_2') +\
                        traci.lane.getLastStepVehicleIDs('m7_3'):

                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_3_positio.csv', 'a') as f1:
            for veh1 in traci.lane.getLastStepVehicleIDs('m4_3') +\
                        traci.lane.getLastStepVehicleIDs('m5_3') +\
                        traci.lane.getLastStepVehicleIDs('m6_3') +\
                        traci.lane.getLastStepVehicleIDs('m7_4'):

                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_4_positio.csv', 'a') as f1:
            for veh1 in traci.lane.getLastStepVehicleIDs('m4_4') +\
                        traci.lane.getLastStepVehicleIDs('m5_4') +\
                        traci.lane.getLastStepVehicleIDs('m6_4') +\
                        traci.lane.getLastStepVehicleIDs('m7_5'):

                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

    def time_space_veh_date(self):
        veh_list_r =list()
        veh_list_0 = list()
        veh_list_1 = list()
        veh_list_2 = list()
        veh_list_3 = list()
        veh_list_4 = list()

        with open(r'space_time_v_lane_r_veh.csv', 'a') as f1:
            veh_list_r += traci.lane.getLastStepVehicleIDs('rin3_0')
            for veh1 in list(set(veh_list_r)):
                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_0_veh.csv', 'a') as f1:

            veh_list_0 += traci.lane.getLastStepVehicleIDs('m4_0')
            for veh1 in list(set(veh_list_0)):
                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_1_veh.csv', 'a') as f1:
            veh_list_1 += traci.lane.getLastStepVehicleIDs('m4_1')
            for veh1 in list(set(veh_list_1)):
                # if traci.vehicle.getDistance(veh1)-239<800:
                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_2_veh.csv', 'a') as f1:

            veh_list_2 += traci.lane.getLastStepVehicleIDs('m4_2')
            for veh1 in list(set(veh_list_2)):
            # if traci.vehicle.getDistance(veh1) - 239 < 800:
                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_3_veh.csv', 'a') as f1:
            veh_list_2 += traci.lane.getLastStepVehicleIDs('m4_3')
            for veh1 in list(set(veh_list_3)):
                # if traci.vehicle.getDistance(veh1)-239<800:
                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

        with open(r'space_time_v_lane_4_veh.csv', 'a') as f1:
            veh_list_2 += traci.lane.getLastStepVehicleIDs('m4_3')
            for veh1 in list(set(veh_list_3)):

                f1.write(veh1)
                f1.write(',')
                f1.write(str(self.sim_step))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getSpeed(veh1), 2)))
                f1.write(',')
                f1.write(str(round(traci.vehicle.getDistance(veh1))))
                f1.write('\n')

    # def policy_NONE(self):
    #     return [Control_Value,Control_Value+2,Control_Value+4,Control_Value+6,Control_Value+8]

    def policy_Static(self):
        return [Control_Value-D_V*2,Control_Value-D_V*1,Control_Value,Control_Value+D_V*1,Control_Value+D_V*2]

    def policy_RB_VSL(self):
        demand_veh = self.traffic_nodes[self.nodes_ids[0]].RB_VSL_parameters()
        print("+++++++++++",self.sim_step,demand_veh)
        dem_num = sum(demand_veh)
        Cd=[35,10]
        if dem_num > Cd[0]:
            speed0 = 10
        elif dem_num <= Cd[0] and dem_num> Cd[1]:
            speed0 = 20
        else:
            speed0 = 30
        return np.full(5, speed0)

    def policy_RB_DVSL(self):
        # speed0 = 30
        #free veh 的数量是交通需求  Cd是bottleneck是通行能力
        demand_veh = self.traffic_nodes[self.nodes_ids[0]].RB_VSL_parameters()
        print("+++++++++++",self.sim_step,demand_veh)
        demand_list = [demand_veh[i] + demand_veh[i + 1] for i in range(len(demand_veh) - 1)]
        Cd=[6,14]
        vsl_s = list()
        for demand_veh in demand_list:
            if demand_veh > Cd[0]:
                vsl_s.append(10)
            elif demand_veh <= Cd[0] and demand_veh> Cd[1]:
                vsl_s.append(20)
            else:
                vsl_s.append(30)
        return vsl_s

class SumoEnvironmentPZ(AECEnv, EzPickle):
    """A wrapper for the SUMO environment that implements the AECEnv interface from PettingZoo.

    For more information, see https://pettingzoo.farama.org/api/aec/.

    The arguments are the same as for :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        """Initialize the environment."""
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEnvironment(**self._kwargs)
        self.agents = self.env.nodes_ids
        self.possible_agents = self.env.nodes_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}
        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.epsido = 0

    def seed(self, seed=None):
        """Set the seed for the environment."""
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.compute_info()
        self.epsido =+ 1

    def compute_info(self):
        """Compute the info for the current step."""
        self.infos = {a: {} for a in self.agents}
        infos = self.env._compute_info()
        for a in self.agents:
            for k, v in infos.items():
                if k.startswith(a):
                    self.infos[a][k] = v

    def observation_space(self, agent):
        """Return the observation space for the agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for the agent."""
        return self.action_spaces[agent]

    def observe(self, agent):
        """Return the observation for the agent."""
        obs = self.env.observations[agent].copy()
        return obs

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        self.save_csv("control",self.epsido)
        self.env.close()

    def render(self):
        """Render the environment."""
        return self.env.render()

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file."""
        self.env.save_csv(out_csv_name, episode)

    def step(self, action):
        """Step the environment."""

        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection

        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "Action for agent {} must be in Discrete({})."
                "It is currently {}".format(agent, self.action_spaces[agent].n, action)
            )

        # action = self.env.policy_RB_VSL()       #policy_RB_DVSL()   #policy_NONE()

        print("action", action)
        # self.env._apply_actions({agent: action})


        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            # # print(self.env._compute_observations(), "++++++++++观测值")
            # for agent, value in self.env.reward_average.items():
            #     # print("value",value)
            #     self.rewards[agent] = sum(value)/self.env.delta_time
            # print(self.env.sim_step, self.rewards, "++++++++++奖励值")
            self.compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]

        self.truncations = {a: done for a in self.agents}
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        print(self.env.sim_step, "++++++++++")

