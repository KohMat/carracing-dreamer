import numpy as np
import torch
from tqdm import trange

from dreamer.carracing import (
    DenseModel,
    Env,
    ObservationDecoder,
    ObservationEncoder,
    Policy,
    Posterior,
    Prior,
)
from dreamer import Dreamer
from dreamer.utils import Buffer, ParallelEnv, Plot

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.backends.cudnn.benchmark = True

    plot = Plot("Itr.", "Score")

    max_itr = 100000
    prefill = 0
    new_episodes = 100
    max_game_frames = 1000
    num_envs = 20
    episode_dir = "episodes"
    weight_dir = "weight"

    env = ParallelEnv([Env() for _ in range(num_envs)])
    agent = Dreamer(
        device=device,
        encoder=ObservationEncoder,
        prior=Prior,
        posterior=Posterior,
        decoder=ObservationDecoder,
        reward=DenseModel,
        policy=Policy,
        value=DenseModel,
    )
    agent.episode_length = 30
    for itr in range(max_itr + prefill):
        scores = []
        for _ in trange(
            new_episodes // num_envs, desc="Collection", leave=False
        ):
            score = 0.0
            buffer = Buffer()
            state = env.reset()
            agent.reset()  # clear the internal state
            for _ in range(max_game_frames):
                action = agent(state, random=True if itr < prefill else False)
                state_, reward, done = env.step(action)

                buffer.add(o=state, a=action, r=reward, done=done)
                score += reward
                state = state_
                env.render()
                if any(done):
                    break

            scores.append(score)

            if buffer.episode_length() >= agent.episode_length:
                buffer.adjust_episode_length(agent.episode_length)
                buffer.dump_each_episode(episode_dir)
            else:
                print(
                    "[Warning]: The agent died before reaching the number of",
                    "frames used for training.",
                    "Discard this episode and proceed to the next.",
                    "Review episode length, number of envs, or env's done.",
                )

        plot.add(itr, np.mean(scores), legend_name="avg", window_name="Score")
        plot.add(itr, np.min(scores), legend_name="min", window_name="Score")
        plot.add(itr, np.max(scores), legend_name="max", window_name="Score")
        print(
            "No. {}\tScore(avg, min, max): {:.2f}, {:.2f}, {:.2f}".format(
                itr, np.mean(scores), np.min(scores), np.max(scores)
            )
        )

        if itr % 25 == 0:
            agent.save_weight(weight_dir)

        if itr > prefill:
            agent.update(episode_dir)
