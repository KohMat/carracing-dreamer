import math

from typing import Union

import gym
import numpy as np
from numpy.lib.npyio import save


class Env:
    def __init__(
        self,
        action_repeat: int = 8,
        img_stack: int = 4,
        save_mp4: Union[str] = None,
    ):
        self.env = gym.make("CarRacing-v0", verbose=0)
        if isinstance(save_mp4, str):
            self.env = gym.wrappers.Monitor(self.env, save_mp4, force=True)
        self.reward_threshold = self.env.spec.reward_threshold
        self.action_repeat = action_repeat
        self.img_stack = img_stack

    def seed(self, seed: int):
        self.env.seed(seed)

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack, dtype=np.float32)[np.newaxis]

    def step(self, action: np.ndarray):
        action = action[0] * np.array([1.0, 0.5, 0.5]) + np.array(
            [0.0, 0.5, 0.5]
        )
        total_reward = 0.0
        for _ in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)

        # # https://github.com/danijar/dreamer/blob/master/dreamer.py#L334
        # total_reward = math.tanh(total_reward)
        #
        scale = 7.0
        total_reward = np.clip(total_reward / scale, -1, 1)

        assert len(self.stack) == self.img_stack
        return (
            np.array(self.stack, dtype=np.float32)[np.newaxis],
            np.array(total_reward, dtype=np.float32)[np.newaxis, np.newaxis],
            np.array(done or die)[np.newaxis, np.newaxis],
        )

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128.0 - 1.0
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 220 steps
        count = 0
        length = 220
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
