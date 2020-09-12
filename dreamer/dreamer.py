from itertools import chain
from math import ceil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributions as td
from visdom import Visdom

from .optimizer import Optimizer
from .rssm import Rssm, State, kl_divergence_between_states
from .utils import Dataset, FreezeParameters, apply


class Dreamer:
    train_steps = 100
    batch_size = 30
    episode_length = 50
    horizon = 15

    deterministic_size = 200
    stochastic_size = 30

    state_dist_type = td.Normal

    free_nats = 3.0
    kl_scale = 1.0

    discount = 0.99
    discount_lambda = 0.95

    visdom = Visdom()

    def __init__(
        self,
        device: torch.device,
        encoder,
        prior,
        posterior,
        decoder,
        reward,
        policy,
        value,
    ):
        self.device = device
        self.encoder = encoder().to(device)
        self.rssm = Rssm(
            prior=prior(
                deterministic_size=self.deterministic_size,
                stochastic_size=self.stochastic_size,
            ).to(device),
            posterior=posterior(
                obs_embed_size=self.encoder.embed_size,
                deterministic_size=self.deterministic_size,
                stochastic_size=self.stochastic_size,
            ).to(device),
            distribution=self.state_dist_type,
        )

        feature_size = self.deterministic_size + self.stochastic_size
        self.decoder = decoder(embed_size=feature_size).to(device)
        self.reward = reward(feature_size, 1).to(device)
        self.policy = policy(feature_size).to(device)
        self.value = value(feature_size, 1).to(device)

        self.optimizer = Optimizer(
            world=[
                self.encoder,
                self.rssm.prior,
                self.rssm.posterior,
                self.decoder,
                self.reward,
            ],
            policy=self.policy,
            value=self.value,
        )

    def __call__(
        self, observation: np.ndarray, train: bool = True, random: bool = False
    ) -> np.ndarray:
        if random:
            low, high = self.policy.explore.action_range
            action = np.random.uniform(
                low, high, (observation.shape[0], self.policy.action_size)
            )
            return action

        o = torch.tensor(observation).float().to(self.device)

        if not hasattr(self, "state"):
            self.state = State.zero(
                o.size(0),
                self.deterministic_size,
                self.stochastic_size,
                self.device,
            )
        if not hasattr(self, "action"):
            self.action = torch.zeros(
                o.size(0),
                self.policy.action_size,
                device=o.device,
                dtype=o.dtype,
            )
        with torch.no_grad():
            embed_o = self.encoder(o)
            prior = self.rssm.predict(self.state, self.action)
            self.state = self.rssm.update(prior, embed_o)
            a = self.policy(self.state.feature())

        if train:
            self.action = self.policy.dist(a).sample()
            self.action = self.policy.explore(self.action)
        else:
            self.action = self.policy.dist(a).mode()

        return self.action.cpu().numpy()

    def reset(self):
        if hasattr(self, "state"):
            delattr(self, "state")
        if hasattr(self, "action"):
            delattr(self, "action")

    def update(self, episode_dir: str):
        dataset = Dataset(episode_dir)
        loader = dataset.loader(batch_size=self.batch_size, shuffle=True)
        repeats = ceil((self.batch_size * self.train_steps) / len(dataset))
        if repeats > 0:
            loaders = [loader] * repeats
            loader = chain(*loaders)

        for step, data in enumerate(loader):
            # the data shape is (time, batch, c, h, w) or (time, batch, 1)
            o = data["o"].float().to(self.device)
            a = data["a"].float().to(self.device)
            r = data["r"].float().to(self.device)
            done = data["done"].float().to(self.device)

            loss_w, posterior = self.world_loss(step, o, a, r, done)
            loss_p, loss_v = self.agent_loss(step, posterior)

            loss = (loss_w.item(), loss_p.item(), loss_v.item())
            print("\r", f"Step: {step}\tLoss(w, p, v): {loss}", end="")

            self.optimizer.zero_grad()
            loss_w.backward()
            loss_p.backward()
            loss_v.backward()
            self.optimizer.clip_grad_norm()
            self.optimizer.step()

            if step >= self.train_steps:
                break

        print()

    def world_loss(
        self,
        step: int,
        o: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, State]:
        embed_o = self.rollout(self.encoder, o)

        posterior = State.zero(
            embed_o.size(1),
            self.deterministic_size,
            self.stochastic_size,
            self.device,
        )

        priors, posteriors = [], []
        for o_i, a_i in zip(embed_o, a):
            prior = self.rssm.predict(posterior, a_i)
            posterior = self.rssm.update(prior, o_i)
            priors.append(prior)
            posteriors.append(posterior)

        features = [p.feature() for p in posteriors]

        prior = State.stack(priors, dim=0)
        posterior = State.stack(posteriors, dim=0)
        div = kl_divergence_between_states(
            prior, posterior, self.free_nats, self.state_dist_type
        )

        recon = self.rollout(self.decoder, features)
        recon_loss = -1.0 * torch.mean(self.decoder.dist(recon).log_prob(o))

        reward = self.rollout(self.reward, features)
        reward_loss = -1.0 * torch.mean(self.reward.dist(reward).log_prob(r))

        loss = self.kl_scale * div + recon_loss + reward_loss

        with torch.no_grad():
            if step >= self.train_steps:
                self.log_imagination(step, o, a, recon, posteriors)

        return loss, posterior

    def rollout(self, network, data):
        x = [network(datum) for datum in data]
        return torch.stack(x, dim=0)

    def agent_loss(
        self, step: int, posterior: State
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def detach_and_reshape(data):
            # exclude the last frame which may contain the done-event
            x = data[:-1].detach().clone()
            # merge the time and batch dimensions
            size = x.size()
            x = torch.reshape(x, (size[0] * size[1], -1))
            return x

        s = apply(posterior, detach_and_reshape)

        with FreezeParameters(self.rssm.prior):
            states = []
            for _ in range(self.horizon):
                a = self.policy(s.detach().feature())
                a = self.policy.dist(a).rsample()
                s = self.rssm.predict(s, a)
                states.append(s)

        features = [s.feature() for s in states]
        feature = torch.stack(features, dim=0)

        with FreezeParameters([self.reward, self.value]):
            # the normal mean is equal to the mode
            reward = self.reward(feature)
            value = self.value(feature)
            discount = self.discount * torch.ones_like(reward)

        returns = self.compute_return(
            reward[:-1],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self.discount_lambda,
        )
        # Make the top row 1 so the cumulative product starts with discount^0
        discount = torch.cat([torch.ones_like(discount[:1]), discount[1:]])
        discount = torch.cumprod(discount[:-1], 0).detach()
        policy_loss = -torch.mean(discount * returns)

        value_feat = feature[:-1].detach()
        v = returns.detach()

        value = self.value(value_feat)
        value = self.value.dist(value)
        log_prob = value.log_prob(v)
        value_loss = -torch.mean(discount * log_prob.unsqueeze(2))

        return policy_loss, value_loss

    def compute_return(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        discount: torch.Tensor,
        bootstrap: torch.Tensor,
        lambda_: float,
    ):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1]
        (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = (
                inp + discount_factor * lambda_ * accumulated_reward
            )
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns

    def log_imagination(
        self,
        step,
        observation,
        action,
        recon,
        posteriors,
        imagination_step: int = 7,
        num_show_episodes: int = 4,
    ):
        s = posteriors[imagination_step - 1]

        def select_episodes(x):
            return x[:num_show_episodes]

        s = apply(s, select_episodes)

        features = []
        for a in action[imagination_step:, :num_show_episodes]:
            s = self.rssm.predict(s, a)
            features.append(s.feature())
        imagination = self.rollout(self.decoder, features)

        inference = torch.cat(
            (recon[:imagination_step, :num_show_episodes], imagination),
            dim=0,
        )

        def make_timeline(data):
            view_idx = list(range(15)) + list(range(15, 50, 5))
            view_idx = list(filter(lambda x: x < data.size(0), view_idx))
            data = [data[i] for i in view_idx]
            data = torch.cat(data, axis=3)
            return data

        inference = make_timeline(inference.cpu())
        observation = make_timeline(observation[:, :num_show_episodes].cpu())

        comps = []
        for o, i in zip(observation, inference):
            comp = torch.cat((o, i), dim=1)
            comps.append(comp)
        comp = torch.cat(comps, dim=1)
        comp = comp[:1]

        comp = (comp - torch.min(comp)) * (
            1 / (torch.max(comp) - torch.min(comp)) * 1.0
        )

        self.visdom.image(
            comp,
            win="RSSM",
            opts=dict(
                title="RSSM",
                caption=f"imagination starts from {imagination_step}",
                store_history=True,
            ),
        )

    def save_weight(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)

        def save(network, filename):
            torch.save(network.state_dict(), filename)

        save(self.encoder, f"{directory}/encoder.pkl")
        save(self.rssm.prior, f"{directory}/prior.pkl")
        save(self.rssm.posterior, f"{directory}/posterior.pkl")
        save(self.decoder, f"{directory}/decoder.pkl")
        save(self.reward, f"{directory}/reward.pkl")
        save(self.policy, f"{directory}/policy.pkl")
        save(self.value, f"{directory}/value.pkl")
        save(self.optimizer, f"{directory}/optimizer.pkl")

    def load_weight(self, directory: str):
        def load(network, filename):
            network.load_state_dict(torch.load(filename))

        load(self.encoder, f"{directory}/encoder.pkl")
        load(self.rssm.prior, f"{directory}/prior.pkl")
        load(self.rssm.posterior, f"{directory}/posterior.pkl")
        load(self.decoder, f"{directory}/decoder.pkl")
        load(self.reward, f"{directory}/reward.pkl")
        load(self.policy, f"{directory}/policy.pkl")
        load(self.value, f"{directory}/value.pkl")
        # load(self.optimizer, f"{directory}/optimizer.pkl")
