import torch
import torch.nn as nn
import pufferlib.spaces


def _padding_from_kernel(kernel_size):
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(k // 2 for k in kernel_size)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        padding = _padding_from_kernel(kernel_size)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.contiguous()
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Conv block expected {self.in_channels} input channels, got {x.shape[1]} "
                f"for tensor shape {tuple(x.shape)}"
            )
        return self.block(x)


class Autoencoder(nn.Module):
    """Standard convolutional autoencoder policy for benchmarking."""

    def __init__(
        self,
        env,
        *args,
        channels=(32, 64, 128, 256, 512),
        hidden_size=512,
        kernel_size=3,
        stride=2,
        channels_last=False,
        downsample=1,
        framestack=None,
        input_size=None,
        output_size=None,
        flat_size=None,
        **kwargs,
    ):
        super().__init__()
        del args, kwargs, input_size, output_size, flat_size
        self.channels_last = channels_last
        self.downsample = downsample
        self.hidden_size = hidden_size
        self._cached_decoded = None

        self.is_multidiscrete = isinstance(
            env.single_action_space, pufferlib.spaces.MultiDiscrete
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            self.num_action_logits = int(sum(self.action_nvec))
        elif self.is_continuous:
            self.num_action_logits = int(env.single_action_space.shape[0])
        else:
            self.num_action_logits = int(env.single_action_space.n)

        obs_shape = env.single_observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError(
                f"Autoencoder expects 3D image observations (H, W, C) or (C, H, W), got {obs_shape}"
            )

        if channels_last:
            h, w, in_channels = obs_shape
        else:
            in_channels, h, w = obs_shape
        self._action_image_hw = (h, w)

        total_pixels = int(h * w)
        if total_pixels < self.num_action_logits:
            raise ValueError(
                "Decoded image does not have enough pixels for action logits. "
                f"Need {self.num_action_logits}, got {total_pixels}."
            )

        if len(channels) < 2:
            raise ValueError("Autoencoder expects at least 2 channel stages.")
        channels = [int(c) for c in channels]
        padding = _padding_from_kernel(kernel_size)

        self.stem = _ConvBlock(in_channels, channels[0], kernel_size=kernel_size)
        self._down_in_channels = []
        self.down_ops = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            self._down_in_channels.append(in_c)
            self.down_ops.append(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            self.enc_blocks.append(_ConvBlock(out_c, out_c, kernel_size=kernel_size))

        self._up_in_channels = []
        self.up_ops = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for in_c, out_c in zip(channels[:0:-1], channels[-2::-1]):
            self._up_in_channels.append(in_c)
            self.up_ops.append(
                nn.ConvTranspose2d(
                    in_c,
                    out_c,
                    kernel_size=stride,
                    stride=stride,
                )
            )
            self.dec_blocks.append(_ConvBlock(out_c, out_c, kernel_size=kernel_size))

        self.action_image_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.action_image_fallback = nn.Linear(hidden_size, h * w)
        self.hidden_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], hidden_size),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        if self.is_continuous:
            self.decoder_logstd = nn.Parameter(torch.zeros(1, self.num_action_logits))

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state=state)

    def encode_observations(self, observations, state=None):
        del state
        x = observations
        if x.ndim == 3:
            if self.channels_last:
                x = x.unsqueeze(-1)
            else:
                x = x.unsqueeze(1)

        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        if self.downsample > 1:
            x = x[:, :, :: self.downsample, :: self.downsample]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        x = x.contiguous()

        x = self.stem(x)
        for expected_in, down, block in zip(self._down_in_channels, self.down_ops, self.enc_blocks):
            if x.shape[1] != expected_in:
                raise ValueError(
                    f"Down conv expected {expected_in} channels, got {x.shape[1]} "
                    f"for tensor shape {tuple(x.shape)}"
                )
            x = block(down(x))
        bottleneck = x

        for expected_in, up, block in zip(self._up_in_channels, self.up_ops, self.dec_blocks):
            if x.shape[1] != expected_in:
                raise ValueError(
                    f"Up conv expected {expected_in} channels, got {x.shape[1]} "
                    f"for tensor shape {tuple(x.shape)}"
                )
            x = up(x)
            x = block(x)

        decoded = x
        self._cached_decoded = decoded
        hidden = self.hidden_proj(bottleneck)
        return hidden

    def decode_actions(self, hidden):
        if (
            self._cached_decoded is not None
            and self._cached_decoded.shape[0] == hidden.shape[0]
        ):
            action_image = self.action_image_head(self._cached_decoded)
        else:
            h, w = self._action_image_hw
            action_pixels_fallback = self.action_image_fallback(hidden)
            action_image = action_pixels_fallback.view(hidden.shape[0], 1, h, w)
        action_pixels = action_image.flatten(start_dim=1)
        if action_pixels.shape[1] < self.num_action_logits:
            raise ValueError(
                "Decoded action image has fewer pixels than required logits. "
                f"Need {self.num_action_logits}, got {action_pixels.shape[1]}."
            )
        action_logits = action_pixels[:, : self.num_action_logits]

        if self.is_multidiscrete:
            logits = action_logits.split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = action_logits
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = action_logits

        values = self.value(hidden)
        return logits, values
