import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

from ..distribution import Distribution
from ..explorer import Explorer


class ObservationEncoder(nn.Module):
    def __init__(
        self, depth=32, stride=2, shape=(4, 96, 96), activation=nn.ReLU
    ):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(shape[0], 1 * depth, 4, stride),
            activation(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride),
            activation(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride),
            activation(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride),
            activation(),
        )
        self.shape = shape
        self.stride = stride
        self.depth = depth

    def forward(self, obs):
        x = self.convolutions(obs)
        x = x.reshape(x.size(0), -1)
        return x

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size


class ObservationDecoder(nn.Module):
    def __init__(
        self,
        embed_size=1024,
        depth=32,
        stride=2,
        activation=nn.ReLU,
        shape=(4, 96, 96),
        dist=Distribution(td.Normal, 3),
    ):
        super().__init__()
        self.depth = depth
        self.shape = shape
        self.dist = dist

        c, h, w = shape
        conv1_kernel_size = 6
        conv2_kernel_size = 6
        conv3_kernel_size = 5
        conv4_kernel_size = 5
        padding = 0
        conv1_shape = conv_out_shape(
            (h, w), padding, conv1_kernel_size, stride
        )
        conv1_pad = output_padding_shape(
            (h, w), conv1_shape, padding, conv1_kernel_size, stride
        )
        conv2_shape = conv_out_shape(
            conv1_shape, padding, conv2_kernel_size, stride
        )
        conv2_pad = output_padding_shape(
            conv1_shape, conv2_shape, padding, conv2_kernel_size, stride
        )
        conv3_shape = conv_out_shape(
            conv2_shape, padding, conv3_kernel_size, stride
        )
        conv3_pad = output_padding_shape(
            conv2_shape, conv3_shape, padding, conv3_kernel_size, stride
        )
        conv4_shape = conv_out_shape(
            conv3_shape, padding, conv4_kernel_size, stride
        )
        conv4_pad = output_padding_shape(
            conv3_shape, conv4_shape, padding, conv4_kernel_size, stride
        )
        self.conv_shape = (32 * depth, *conv4_shape)
        self.linear = nn.Linear(
            embed_size, 32 * depth * np.prod(conv4_shape).item()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32 * depth,
                4 * depth,
                conv4_kernel_size,
                stride,
                output_padding=conv4_pad,
            ),
            activation(),
            nn.ConvTranspose2d(
                4 * depth,
                2 * depth,
                conv3_kernel_size,
                stride,
                output_padding=conv3_pad,
            ),
            activation(),
            nn.ConvTranspose2d(
                2 * depth,
                1 * depth,
                conv2_kernel_size,
                stride,
                output_padding=conv2_pad,
            ),
            activation(),
            nn.ConvTranspose2d(
                1 * depth,
                shape[0],
                conv1_kernel_size,
                stride,
                output_padding=conv1_pad,
            ),
        )

    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (-1, *self.conv_shape))
        x = self.decoder(x)
        return x


class Prior(nn.Module):
    def __init__(
        self,
        action_size=3,
        stochastic_size=30,
        deterministic_size=200,
        hidden_size=200,
        activation=nn.ELU,
    ):
        super().__init__()
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._cell = nn.GRUCell(hidden_size, deterministic_size)
        self._rnn_input_model = self._build_rnn_input_model()
        self._stochastic_prior_model = self._build_stochastic_model()

    def _build_rnn_input_model(self):
        rnn_input_model = [
            nn.Linear(self._action_size + self._stoch_size, self._hidden_size)
        ]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(self._hidden_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [
            nn.Linear(self._hidden_size, 2 * self._stoch_size)
        ]
        return nn.Sequential(*stochastic_model)

    def forward(
        self,
        deterministic: torch.Tensor,
        stochastic: torch.Tensor,
        action: torch.Tensor,
    ):
        rnn_input = self._rnn_input_model(
            torch.cat([action, stochastic], dim=-1)
        )
        deter_state = self._cell(rnn_input, deterministic)
        mean, stddev = torch.chunk(
            self._stochastic_prior_model(deter_state), 2, dim=-1
        )
        stddev = F.softplus(stddev) + 0.1
        return deter_state, mean, stddev


class Posterior(nn.Module):
    def __init__(
        self,
        obs_embed_size,
        stochastic_size=30,
        deterministic_size=200,
        hidden_size=200,
        activation=nn.ELU,
    ):
        super().__init__()
        self._obs_embed_size = obs_embed_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._stochastic_posterior_model = self._build_stochastic_model()

    def _build_stochastic_model(self):
        stochastic_model = [
            nn.Linear(
                self._deter_size + self._obs_embed_size, self._hidden_size
            )
        ]
        stochastic_model += [self._activation()]
        stochastic_model += [
            nn.Linear(self._hidden_size, 2 * self._stoch_size)
        ]
        return nn.Sequential(*stochastic_model)

    def forward(
        self,
        deterministic: torch.Tensor,
        obs_embed: torch.Tensor,
    ):
        x = torch.cat([deterministic, obs_embed], -1)
        mean, stddev = torch.chunk(
            self._stochastic_posterior_model(x), 2, dim=-1
        )
        stddev = F.softplus(stddev) + 0.1
        return mean, stddev


class DenseModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layers: int = 3,
        hidden_size: int = 300,
        activation=nn.ELU,
        dist=Distribution(td.Normal, 1),
    ):
        super().__init__()
        self._output_size = output_size
        self._layers = layers
        self._hidden_size = hidden_size
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._input_size = input_size
        # Defining the structure of the NN
        self.model = self.build_model()
        self.dist = dist

    def build_model(self):
        model = [nn.Linear(self._input_size, self._hidden_size)]
        model += [self.activation()]
        for _ in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, self._output_size)]
        return nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Policy(nn.Module):
    def __init__(
        self,
        input_size,
        action_size=3,
        hidden_size=200,
        layers=3,
        activation=nn.ELU,
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
        dist=Distribution(
            td.Normal, 1, transform=td.transforms.TanhTransform()
        ),
        explorer=Explorer(action_range=(-1, 1)),
        mean_act=torch.tanh,
    ):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.activation = activation
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.feedforward_model = self.build_model()
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)
        self.dist = dist
        self.explore = explorer
        self.mean_act = mean_act

    def build_model(self):
        model = [nn.Linear(self.input_size, self.hidden_size)]
        model += [self.activation()]
        for _ in range(1, self.layers):
            model += [nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self.hidden_size, self.action_size * 2)]
        return nn.Sequential(*model)

    def forward(self, feature):
        x = self.feedforward_model(feature)
        mean, std = torch.chunk(x, 2, -1)
        mean = self.mean_scale * self.mean_act(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        return mean, std


def conv_out(h_in, padding, kernel_size, stride):
    return int(
        (h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0
    )


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(
        output_padding(h_in[i], conv_out[i], padding, kernel_size, stride)
        for i in range(len(h_in))
    )
