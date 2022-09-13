import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path


class ACC6Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.th = 10
        # self.viewer = None

        high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x1, x2, x3, x4, x5, x6 = self.state
        done = False
        u1 = np.clip(u, -self.th, self.th)[0]
        u2 = np.clip(u, -1, 1)[1]

        t = 0.02
        x1_new = x1 + x2 * t
        x2_new = x2 + x3 * t
        x3_new = x3 + (2 * u1 - 2 * x3 - x2 * x2 / 10000) * t
        x4_new = x4 + x5 * t
        x5_new = x5 + x6 * t
        x6_new = x6 + (2 * u2 - 2 * x6 - x5 * x5 / 10000) * t

        self.state = np.array([x1_new, x2_new, x3_new, x4_new, x5_new, x6_new], dtype=np.float32)

        reward = - (abs(x2_new - 22.84) + abs(x5_new - 29.9))
        # reward = -abs(x4_new - 29.9)
        # reward = - abs(x2_new - 22.84)
        # if 22.87 >= x1_new >= 22.81:
        #     reward += 1000
        # if 30.02 >= x4_new >= 29.88:
        #     reward += 1000
        # reward = -1 - abs(x2_new - 29.9)
        reward = -1
        if 22.87 >= x2_new >= 22.81 and 30.02 >= x5_new >= 29.88:
            # if 30.02 >= x5_new >= 29.88:
            # if 22.87 >= x2_new >= 22.81:
            reward = 1000
            done = True

        return self._get_obs(), reward, done, {}

    def reset(self):
        high = np.array([91, 32.05, 0, 11, 30.05, 0])
        low = np.array([90, 32, 0, 10, 30, 0])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def _get_obs(self):
        self.state[0] = np.clip(self.state[0], 0, 199)
        self.state[1] = np.clip(self.state[1], -99, 99)
        self.state[2] = np.clip(self.state[2], -99, 99)
        self.state[3] = np.clip(self.state[3], 0, 199)
        self.state[4] = np.clip(self.state[4], -99, 99)
        self.state[5] = np.clip(self.state[5], -99, 99)
        return self.state
