"""Policy network for the DASM environment.

Standalone policy module that can be modified independently of PufferLib's
built-in policies. The architecture mirrors pufferlib.models.Default but
lives here so you can iterate on it without touching the library.
"""

import numpy as np
import torch
import torch.nn as nn

import pufferlib.pytorch
import pufferlib.spaces


class Policy(nn.Module):
    """MLP policy for DASM continuous control.

    observation -> encoder (Linear + GELU) -> actor_head / critic_head
    Continuous actions use a learned mean + log-std parameterization.
    """

    def __init__(self, env, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = isinstance(
            env.single_action_space, pufferlib.spaces.Box
        )

        num_obs = int(np.prod(env.single_observation_space.shape))
        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
            nn.GELU(),
        )

        if self.is_continuous:
            act_dim = env.single_action_space.shape[0]
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, act_dim), std=0.01
            )
            self.decoder_logstd = nn.Parameter(torch.zeros(1, act_dim))
        else:
            num_actions = env.single_action_space.n
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_actions), std=0.01
            )

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1
        )

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1).float()
        return self.encoder(observations)

    def decode_actions(self, hidden):
        if self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values
