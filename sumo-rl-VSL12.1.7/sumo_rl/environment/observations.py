"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces


from .traffic_node import TrafficNode

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, node: TrafficNode):
        """Initialize observation function."""
        self.node = node

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self,node: TrafficNode):
        """Initialize default observation function."""
        super().__init__(node)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        state_occ =  self.node._observation_fn_default()  # one-hot encoding
        simulation_step = self.node.sumo.simulation.getTime()/self.node.env.sim_max_time
        state_occ = np.append(state_occ,simulation_step)
        observation = np.array(state_occ, dtype=np.float32)
        # print(len(observation),observation,"++++++++++++++++++++++")
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        # print("状态空间", self.node.num_state())
        return spaces.Box(
            low=0, high=2, shape=(45,), dtype=np.float32)


