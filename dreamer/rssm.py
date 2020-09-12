from typing import NamedTuple

import torch
import torch.distributions as td


class State(NamedTuple):
    mean: torch.Tensor
    stddev: torch.Tensor
    deterministic: torch.Tensor
    stochastic: torch.Tensor

    def feature(self) -> torch.Tensor:
        return torch.cat((self.stochastic, self.deterministic), dim=-1)

    def distribution(self, dist_type):
        hidden_state_dims = 1
        return td.independent.Independent(
            dist_type(self.mean, self.stddev), hidden_state_dims
        )

    def detach(self):
        return State(
            self.mean.detach(),
            self.stddev.detach(),
            self.deterministic.detach(),
            self.stochastic.detach(),
        )

    @classmethod
    def zero(
        self,
        batch_size: int,
        deterministic_size: int,
        stochastic_size: int,
        device: torch.device = None,
    ):
        deterministic_shape = (batch_size, deterministic_size)
        stochastic_shape = (batch_size, stochastic_size)
        if device is not None:
            return State(
                torch.zeros(stochastic_shape, device=device),
                torch.zeros(stochastic_shape, device=device),
                torch.zeros(deterministic_shape, device=device),
                torch.zeros(stochastic_shape, device=device),
            )
        else:
            return State(
                torch.zeros(stochastic_shape),
                torch.zeros(stochastic_shape),
                torch.zeros(deterministic_shape),
                torch.zeros(stochastic_shape),
            )

    @classmethod
    def stack(self, states: list, dim=0):
        return State(
            torch.stack([state.mean for state in states], dim=dim),
            torch.stack([state.stddev for state in states], dim=dim),
            torch.stack([state.deterministic for state in states], dim=dim),
            torch.stack([state.stochastic for state in states], dim=dim),
        )

    @classmethod
    def from_tuple(self, data):
        return State(*data)


class Rssm:
    def __init__(
        self,
        prior,
        posterior,
        distribution=td.Normal,
    ):
        self.prior = prior
        self.posterior = posterior
        self.distribution = distribution

    def predict(self, s: State, a: torch.Tensor) -> State:
        deterministic, mean, stddev = self.prior(
            s.deterministic, s.stochastic, a
        )
        hidden_state_dims = 1
        dist = td.independent.Independent(
            self.distribution(mean, stddev), hidden_state_dims
        )
        return State(mean, stddev, deterministic, dist.rsample())

    def update(self, s: State, o: torch.Tensor) -> State:
        mean, stddev = self.posterior(s.deterministic, o)
        hidden_state_dims = 1
        dist = td.independent.Independent(
            self.distribution(mean, stddev), hidden_state_dims
        )
        return State(mean, stddev, s.deterministic, dist.rsample())


def kl_divergence_between_states(
    prior: State, posterior: State, min_nats: float, dist_type
) -> torch.Tensor:
    prior_dist = prior.distribution(dist_type)
    posterior_dist = posterior.distribution(dist_type)

    # Care the order args of the kl_divergence, since
    # div(q,p) != div(p, q)
    # the Kullback–Leibler divergence from Q to P is defined to be Div(P|Q
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    # See the rssm impl.
    # https://github.com/danijar/dreamer/blob/master/dreamer.py#L174
    div = torch.mean(
        torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
    )
    div = torch.max(
        div,
        div.new_full(div.size(), min_nats),
    )
    return div
