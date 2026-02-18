import pufferlib.models
from pufferlib.environments.policy_wrappers import (
    Autoencoder as SharedAutoencoder,
    UNet as SharedUNet,
)


class MLP(pufferlib.models.Default):
    default_channels_last = True


Policy = MLP
NoPolicy = pufferlib.models.NoPolicy

class Autoencoder(SharedAutoencoder):
    default_channels_last = True


class UNet(SharedUNet):
    default_channels_last = True
