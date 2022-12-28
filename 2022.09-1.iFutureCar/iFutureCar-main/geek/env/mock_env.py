from pathlib import Path
from typing import Dict, List, Optional

import gym
import numpy as np

import dill
from geek.env.matrix_env import DoneReason, MatrixEnv, Observations, Scenarios


class MockEnv(MatrixEnv):
    def __init__(self, scenarios: Scenarios, render_id: Optional[str]=None) -> None:
        if scenarios is None:
            raise TypeError("scenarios should not be none.")

        self.obs_index_ = 0
        self.scenarios_: Scenarios = scenarios
        self.obs_: List[Observations] = []
        self._init_obs_data()

    def _init_obs_data(self) -> None:
        obs_data_file = Path(__file__).absolute().parent.joinpath("obs_data.pickle")
        if not obs_data_file.exists():
            raise FileNotFoundError(
                f"mock action data: {obs_data_file} not exists, "
                f"and please prepare mock data before run MockEnv."
            )

        with open(obs_data_file, "rb") as file:
            dill.detect.trace(True)
            self.obs_ = dill.load(file)

        assert len(self.obs_), "load mock observation data failed."

    def reset(self) -> Dict:
        self.obs_index_ = 0
        
        observation, _, _, _ = self.obs_[self.obs_index_]
        return observation

    def step(self, actions: np.ndarray) -> Observations:
        self.obs_index_ += 1
        if self.obs_index_ >= len(self.obs_):
            raise RuntimeError(f"reset when done")
        
        obs = self.obs_[self.obs_index_]
        return obs

    def instance_key(self) -> str:
        return "mock key"


gym.envs.register(id="MatrixEnv-v1", entry_point="geek.env.mock_env:MockEnv")
