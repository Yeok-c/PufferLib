import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib.spaces


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(k // 2 for k in kernel_size)
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


class UNet(nn.Module):
    """UNet++ style policy used for inference benchmarking."""

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
                f"UNet expects 3D image observations (H, W, C) or (C, H, W), got {obs_shape}"
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

        if len(channels) != 5:
            raise ValueError(
                "UNet currently expects exactly 5 channel stages. "
                f"Got {len(channels)}: {channels}"
            )

        c0, c1, c2, c3, c4 = [int(c) for c in channels]
        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        self._pool_stride = int(stride) if isinstance(stride, int) else int(stride[0])

        # Encoder
        self.x0_0 = _ConvBlock(in_channels, c0, kernel_size=kernel_size)
        self.x1_0 = _ConvBlock(c0, c1, kernel_size=kernel_size)
        self.x2_0 = _ConvBlock(c1, c2, kernel_size=kernel_size)
        self.x3_0 = _ConvBlock(c2, c3, kernel_size=kernel_size)
        self.x4_0 = _ConvBlock(c3, c4, kernel_size=kernel_size)

        # UNet++ nested decoder
        self.x0_1 = _ConvBlock(c0 + c1, c0, kernel_size=kernel_size)
        self.x1_1 = _ConvBlock(c1 + c2, c1, kernel_size=kernel_size)
        self.x2_1 = _ConvBlock(c2 + c3, c2, kernel_size=kernel_size)
        self.x3_1 = _ConvBlock(c3 + c4, c3, kernel_size=kernel_size)

        self.x0_2 = _ConvBlock(c0 * 2 + c1, c0, kernel_size=kernel_size)
        self.x1_2 = _ConvBlock(c1 * 2 + c2, c1, kernel_size=kernel_size)
        self.x2_2 = _ConvBlock(c2 * 2 + c3, c2, kernel_size=kernel_size)

        self.x0_3 = _ConvBlock(c0 * 3 + c1, c0, kernel_size=kernel_size)
        self.x1_3 = _ConvBlock(c1 * 3 + c2, c1, kernel_size=kernel_size)

        self.x0_4 = _ConvBlock(c0 * 4 + c1, c0, kernel_size=kernel_size)

        # Wasteful action image decoder: we flatten and consume first N pixels as logits.
        self.action_image_head = nn.Conv2d(c0, 1, kernel_size=1)
        self.action_image_fallback = nn.Linear(hidden_size, h * w)
        self.hidden_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        if self.is_continuous:
            self.decoder_logstd = nn.Parameter(torch.zeros(1, self.num_action_logits))

    @staticmethod
    def _upsample_like(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _safe_pool(self, x):
        # Tiny observations (e.g., Minigrid) can hit 1x1 feature maps early.
        # Stop downsampling instead of pooling to 0x0.
        if min(x.shape[-2:]) < self._pool_stride:
            return x
        return self.pool(x)

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

        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self._safe_pool(x0_0))
        x2_0 = self.x2_0(self._safe_pool(x1_0))
        x3_0 = self.x3_0(self._safe_pool(x2_0))
        x4_0 = self.x4_0(self._safe_pool(x3_0))

        x0_1 = self.x0_1(torch.cat([x0_0, self._upsample_like(x1_0, x0_0)], dim=1))
        x1_1 = self.x1_1(torch.cat([x1_0, self._upsample_like(x2_0, x1_0)], dim=1))
        x2_1 = self.x2_1(torch.cat([x2_0, self._upsample_like(x3_0, x2_0)], dim=1))
        x3_1 = self.x3_1(torch.cat([x3_0, self._upsample_like(x4_0, x3_0)], dim=1))

        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self._upsample_like(x1_1, x0_0)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self._upsample_like(x2_1, x1_0)], dim=1))
        x2_2 = self.x2_2(torch.cat([x2_0, x2_1, self._upsample_like(x3_1, x2_0)], dim=1))

        x0_3 = self.x0_3(
            torch.cat([x0_0, x0_1, x0_2, self._upsample_like(x1_2, x0_0)], dim=1)
        )
        x1_3 = self.x1_3(
            torch.cat([x1_0, x1_1, x1_2, self._upsample_like(x2_2, x1_0)], dim=1)
        )

        x0_4 = self.x0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample_like(x1_3, x0_0)], dim=1)
        )

        self._cached_decoded = x0_4
        hidden = self.hidden_proj(x4_0)
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
