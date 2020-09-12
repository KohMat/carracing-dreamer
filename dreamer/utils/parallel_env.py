from typing import Tuple

import numpy as np


class ParallelEnv:
    def __init__(self, envs):
        self.env = envs
        self.num_envs = len(envs)
        self.seed(0)

    def seed(self, seed: int):
        [env.seed(seed + idx) for idx, env in enumerate(self.env)]

    def reset(self) -> np.ndarray:
        s = [env.reset() for env in self.env]
        s = np.concatenate(s, axis=0)
        return s

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, r, done = [], [], []
        for env, a in zip(self.env, action):
            x = env.step(a[np.newaxis])
            s.append(x[0])
            r.append(x[1])
            done.append(x[2])

        s = np.concatenate(s, axis=0)
        r = np.concatenate(r, axis=0)
        done = np.concatenate(done, axis=0)
        return s, r, done

    def render(self, index: int = 0):
        self.env[index].render()
