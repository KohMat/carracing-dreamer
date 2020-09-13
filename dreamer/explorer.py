from typing import Optional, Tuple

import torch
import torch.distributions as td

from .distribution import Distribution


class Explorer:
    def __init__(
        self,
        noise: float = 0.3,
        action_range: Tuple[int, int] = (-1, 1),
        dims=1,
        expl_min: float = 0.0,
        expl_decay: float = 0.0,
    ):
        self.noise = noise
        self.action_range = action_range
        if self.action_range[1] < self.action_range[0]:
            raise ValueError(
                "The min of the action_range should be higher than the max"
            )
        self.dist = Distribution(td.Normal, dims=dims)
        self.expl_min = expl_min
        self.expl_decay = expl_decay

    def __call__(
        self, action: torch.Tensor, step: Optional[int] = None
    ) -> torch.Tensor:
        amount = self.noise
        if self.expl_decay and step:  # Linear decay
            amount -= step / self.expl_decay

        if self.expl_min:
            amount = max(self.expl_min, amount)

        return torch.clamp(
            self.dist((action, amount)).sample(),
            min=self.action_range[0],
            max=self.action_range[1],
        )
