from typing import Optional, Tuple

import torch


class Explorer:
    def __init__(
        self,
        noise: float = 0.1,
        action_range: Tuple[int, int] = (-1, 1),
        expl_type: str = "additive_gaussian",
        expl_min: float = 0.00,
        expl_decay: int = 7000,
    ):
        self.noise = noise
        self.action_range = action_range
        if self.action_range[1] < self.action_range[0]:
            raise ValueError(
                "The min of the action_range should be higher than the max"
            )
        self.expl_type = expl_type
        self.expl_min = expl_min
        self.expl_decay = expl_decay

    def __call__(
        self, action: torch.Tensor, step: Optional[int] = None
    ) -> torch.Tensor:
        """
        :param action: action to take,
        shape (1,) (if categorical), or (action dim,) (if continuous)
        :return: action of the same shape passed in, augmented with some noise
        """
        amount = self.noise
        if self.expl_decay and step:  # Linear decay
            amount -= step / self.expl_decay

        if self.expl_min:
            amount = max(self.expl_min, amount)

        if self.expl_type == "additive_gaussian":  # For continuous actions
            scale = (self.action_range[1] - self.action_range[0]) / 2
            noise = (
                scale
                * amount
                * torch.randn(*action.shape, device=action.device)
            )
            return torch.clamp(
                action + noise,
                min=self.action_range[0],
                max=self.action_range[1],
            )

        raise NotImplementedError(self.expl_type)
