import numpy as np
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper

import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class arcwrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Dict({
                # **env.observation_space,
                'grid': Box(0, 10, (5, 5), dtype=np.uint8), 
                'grid_dim': Box(1, 5, (2,), dtype=np.uint8), 
                'input': Box(0, 10, (5, 5), dtype=np.uint8), 
                'input_dim': Box(1, 5, (2,), dtype=np.uint8),
                "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        }
        )
        #self.observation_space['grid'] = gym.spaces.Box(0, 10, (30,30,1), dtype = np.uint8)


    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = {
            **obs,
            "is_first": False,
            "is_last": done,
            "is_terminal": done,
        }
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = {
            **obs,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }

        return obs, info