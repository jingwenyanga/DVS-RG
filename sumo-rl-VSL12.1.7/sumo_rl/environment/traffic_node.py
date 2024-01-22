"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
import math
from typing import Callable, List, Union

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces
import gym

Net_info = {"304822600": {
    'in_main': ['m4'],
    'vsl_main': ['m5'],
    'free_main': ['m6'],
    'merg_main': ['m7'],
    'in_ramp': ['rin3'],
    'out_ramp': ['rout1'],
    'out_main': ['m8'],
    'merging_det': ['em70', 'em71','em72','em73','em74', 'em75'],
    'inflow_det': ['em40', 'em41', 'em42', 'em43', 'em44', 'erin20'],
    'outflow_det': ['em80', 'em81', 'em82', 'em83', 'em84', 'erout10']
}}



class TrafficNode:
    """This class represents a Traffic Node controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(
            self,
            env,
            node_id: str,

            reward_fn: Union[str, Callable],
            sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = node_id
        self.env = env
        self.seg_lengths = 200  # 每个路段的长度
        self.reward_fn = reward_fn
        self.sumo = sumo
        # self.next_action_time = begin_time

        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficNode.reward_fns.keys():
                self.reward_fn = TrafficNode.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")

        self.observation_fn = self.env.observation_class(self)


        self.in_main_edge = Net_info[self.id]["in_main"][0]
        self.vsl_main_edge = Net_info[self.id]["vsl_main"][0]
        self.free_main_edge = Net_info[self.id]["free_main"][0]
        self.merg_main_edge = Net_info[self.id]["merg_main"][0]
        self.out_main_edge = Net_info[self.id]["out_main"][0]

        self.in_ramp_edge = Net_info[self.id]["in_ramp"][0]
        self.out_ramp_edge = Net_info[self.id]["out_ramp"][0]

        self.merging_det = Net_info[self.id]["merging_det"]
        self.inflow_det = Net_info[self.id]["inflow_det"]
        self.outflow_det = Net_info[self.id]["outflow_det"]

        self.edges = self.all_edges()
        self.lanes = self.all_lanes()
        self.lanes_lenght = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes}
        self.maxSpeed = 30 #m/s

        self.observation_space = self.observation_fn.observation_space()
        self.vsl_lane = self.vsl_lanes()


    def all_edges(self):
        all_edge = [self.in_main_edge,self.vsl_main_edge,self.free_main_edge,self.merg_main_edge,self.in_ramp_edge,self.out_ramp_edge,self.out_main_edge]
        return all_edge

    def all_lanes(self):
        all_lane = list()
        edge_list = self.edges
        for j in edge_list:
            num_lane = self.sumo.edge.getLaneNumber(j)
            for i in range(num_lane):
                laneid = j + '_' + str(i)
                all_lane.append(laneid)
        return all_lane

    def state_lanes(self):
        state_lane = list()
        state_edge = [self.in_main_edge, self.vsl_main_edge, self.free_main_edge, self.merg_main_edge, self.in_ramp_edge]
        for j in state_edge:
            num_lane = self.sumo.edge.getLaneNumber(j)
            for i in range(num_lane):
                laneid = j + '_' + str(i)
                state_lane.append(laneid)
        return state_lane


    def vsl_lanes(self):
        vsl_lane = list()
        num_lane = self.sumo.edge.getLaneNumber(self.vsl_main_edge)
        for i in range(num_lane):
            laneid = self.vsl_main_edge + '_' + str(i)
            vsl_lane.append(laneid)
        return vsl_lane

    @property
    def action_space(self):
        return spaces.Box(
            low=np.float32(0), high=np.float32(8), shape=(len(self.vsl_lane),), dtype=np.float32
        )

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        self.last_reward = self.reward_fn(self)
        return self.last_reward

    def _get_veh_list(self):
        veh_list = []
        for edge in self.edges:
            veh_list += self.sumo.edge.getLastStepVehicleIDs(edge)
        return veh_list


    def get_VSL_area_speed(self):
        VSL_lanes = list()
        num_lane = self.sumo.edge.getLaneNumber(self.vsl_main_edge)
        for i in range(num_lane):
            laneid = self.vsl_main_edge + '_' + str(i)
            VSL_lanes.append(laneid)
        veh_info = list()
        for lane in VSL_lanes:
            veh_s = list()
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                veh_s.append(self.sumo.vehicle.getSpeed(veh))
            veh_info.append(veh_s)
        return [np.mean(ave_s) for ave_s in veh_info]

    def get_BT_merging_lane(self):
        merg_lanes = list()
        num_lane = self.sumo.edge.getLaneNumber(self.merg_main_edge)
        for i in range(num_lane):
            laneid = self.merg_main_edge + '_' + str(i)
            merg_lanes.append(laneid)
        veh_num = list()
        for lane in merg_lanes:
            veh_num.append(len(self.sumo.lane.getLastStepVehicleIDs(lane)))
        return veh_num

    """_get_per_agent_info"""
    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.
        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        speed_list = list()
        for lane in self.lanes:
            speed_list.append(self.sumo.lane.getLastStepMeanSpeed(lane))
        return np.mean(speed_list)

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_out_flow(self):
        return [self.sumo.inductionloop.getLastIntervalVehicleNumber(i) for i in self.outflow_det]

    def get_in_flow(self):
        return [self.sumo.inductionloop.getLastIntervalVehicleNumber(i) for i in self.inflow_det]

    def get_merging_flow(self):
        return [self.sumo.inductionloop.getLastIntervalVehicleNumber(i) for i in self.merging_det]

    def RB_VSL_parameters(self):
        onramp_free_lanes = list()
        in_ramp_edge = self.in_ramp_edge
        merg_main_edge = self.merg_main_edge
        for j in [in_ramp_edge,merg_main_edge]:
            num_lane = self.sumo.edge.getLaneNumber(j)
            for i in range(num_lane):
                laneid = j + '_' + str(i)
                onramp_free_lanes.append(laneid)
        veh_num = list()
        for lane in onramp_free_lanes:
            veh_num.append(len(self.sumo.lane.getLastStepVehicleIDs(lane)))
        return veh_num



    #####################  the bottleneck speed ####################
    def calc_bottlespeed(self):
        return self.sumo.edge.getLastStepMeanSpeed(self.merg_main_edge)

    def calc_bottleoccputation(self):
        return self.sumo.edge.getLastStepOccupancy(self.merg_main_edge)

    #####################  the number of emergency braking vehicles ####################
    def calc_halt_veh(self):
        """Returns the total number of halting vehicles for the last time step on the given edge.
        A speed of less than 0.1 m/s is considered a halt."""
        halt_veh= list()
        for i in self.edges:
            halt_veh.append(self.sumo.edge.getLastStepHaltingNumber(i))
        return np.sum(halt_veh)

    #####################  the CO, NOx, HC, PMx emission  ####################
    def calc_emission(self):
        vidlist = self.edges
        co = []
        hc = []
        nox = []
        pmx = []
        for vid in vidlist:
            co.append(self.sumo.edge.getCOEmission(vid))
            hc.append(self.sumo.edge.getHCEmission(vid))
            nox.append(self.sumo.edge.getNOxEmission(vid))
            pmx.append(self.sumo.edge.getPMxEmission(vid))
        return np.sum(np.self.sumo(co)), np.sum(np.array(hc)), np.sum(np.array(nox)), np.sum(np.array(pmx))

    def calc_r_jam(self):
        return self.sumo.edge.getLastStepHaltingNumber(self.in_ramp_edge)
               #/self.sumo.edge.getLastStepMeanSpeed(self.in_ramp_edge)


    def _calc_outflow(self):
        return sum(self.get_out_flow())-sum(self.get_in_flow())

    def _calc_bottlespeed(self):
        return self.calc_bottlespeed()

    def _calc_numraking_veh(self):
        pass

    def _calc_emission(self):
        co, hc, nox, pmx = self.calc_emission()
        return -(co / 1.5 + hc / 0.13 + nox / 0.04 + pmx / 0.01)


    def _bott_speed_rjam(self):
        return self.calc_bottlespeed()-self.calc_r_jam()




    def num_state(self):
        return len(self.all_lanes())

    def _observation_fn_default(self):
        observation = []
        all_lane = self.state_lanes()
        for i in all_lane:
            # print(i,self.sumo.lane.getLastStepVehicleIDs(i))
            observation.append(self.sumo.lane.getLastStepOccupancy(i))
            observation.append(self.sumo.lane.getLastStepMeanSpeed(i)/self.maxSpeed)
        # print("observation", len(observation), observation)

        return np.array(observation, dtype=np.float32)

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "cal_in_outflow": _calc_outflow,
        "calc_bottlespeed": _calc_bottlespeed,
        "bott_speed_rjam": _bott_speed_rjam,
        "cal_bark": _calc_numraking_veh,
        "cal_emission": _calc_emission,
    }
