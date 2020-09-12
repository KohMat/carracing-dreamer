from typing import Iterable, List, Tuple, Union

import torch
import torch.nn as nn


class Optimizer:
    world_lr = 6e-4
    value_lr = 8e-5
    policy_lr = 8e-5
    grad_clip = 100.0

    def __init__(self, world, policy, value, optimizer_state: dict = None):
        self.world_params = self.parameters(world)
        self.policy_params = self.parameters(policy)
        self.value_params = self.parameters(value)

        self.world = torch.optim.Adam(self.world_params, lr=self.world_lr)
        self.policy = torch.optim.Adam(self.policy_params, lr=self.policy_lr)
        self.value = torch.optim.Adam(self.value_params, lr=self.value_lr)

        if optimizer_state is not None:
            self.load_state(optimizer_state)

    def zero_grad(self):
        self.world.zero_grad()
        self.policy.zero_grad()
        self.value.zero_grad()

    def step(self):
        self.world.step()
        self.policy.step()
        self.value.step()

    def clip_grad_norm(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_norm_world = nn.utils.clip_grad_norm_(
            self.world_params, self.grad_clip
        )
        grad_norm_actor = nn.utils.clip_grad_norm_(
            self.policy_params, self.grad_clip
        )
        grad_norm_value = nn.utils.clip_grad_norm_(
            self.value_params, self.grad_clip
        )
        return (grad_norm_world, grad_norm_actor, grad_norm_value)

    @classmethod
    def parameters(
        self, module: Union[nn.Module, Iterable[nn.Module]]
    ) -> List[nn.Parameter]:
        """
        Given torch modules, returns a list of their parameters.
        :param modules: iterable of modules
        :returns: a list of parameters
        """
        if isinstance(module, nn.Module):
            return list(module.parameters())

        elif isinstance(module, (list, tuple)):
            world_parameters = []
            for m in module:
                world_parameters += list(m.parameters())
            return world_parameters

        elif isinstance(module, dict):
            world_parameters = []
            for k in module.keys():
                world_parameters += list(module[k].parameters())
            return world_parameters

        else:
            raise ValueError("The given type is not supported.")

    def state_dict(self):
        """Return the optimizer state dict (e.g. Adam); overwrite if using
        multiple optimizers."""
        return dict(
            world=self.world.state_dict(),
            policy=self.policy.state_dict(),
            value=self.value.state_dict(),
        )

    def load_state_dict(self, state_dict):
        """Load an optimizer state dict; should expect the format returned
        from ``optim_state_dict().``"""
        self.world.load_state_dict(state_dict["world"])
        self.policy.load_state_dict(state_dict["policy"])
        self.value.load_state_dict(state_dict["value"])
