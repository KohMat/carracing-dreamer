import torch
import torch.distributions as td
import torch.nn.functional as F


class Distribution:
    def __init__(self, dist=td.Normal, dims=1, transform=None):
        self.dist = dist
        self.dims = dims
        self.transform = transform

    def __call__(self, x):
        if self.dist == td.Normal or self.dist == td.Beta:
            if isinstance(x, torch.Tensor):
                dist = self.dist(x, 1.0)
            elif isinstance(x, tuple):
                dist = self.dist(x[0], x[1])
            else:
                raise ValueError("The dist type is not supported.")
        else:
            raise ValueError("The dist type is not supported.")

        if self.transform is not None:
            dist = torch.distributions.TransformedDistribution(
                dist, self.transform
            )
        dist = torch.distributions.Independent(dist, self.dims)
        dist = SampleDist(dist)
        return dist


class SampleDist:
    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def sample(self):
        return self._dist.sample()
