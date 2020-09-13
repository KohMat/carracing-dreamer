import torch

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

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    weight_dir = "weight"

    env = Env(save_mp4="video")
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
    agent.load_weight(weight_dir)

    score = 0.0
    state = env.reset()
    while 1:
        action = agent(state, train=False)
        state_, reward, done = env.step(action)

        score += reward
        state = state_
        env.render()
        if any(done):
            break

    print("Score:", score)
