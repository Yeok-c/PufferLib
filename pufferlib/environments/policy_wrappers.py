import pufferlib.models
import pufferlib.pytorch
import pufferlib.spaces

IMAGE_DIMS = 3
BATCH_FLAT_DIMS = 2


class _SpaceProxy:
    def __init__(self, shape):
        self.shape = shape


class _EnvProxy:
    def __init__(self, env, obs_shape):
        self._env = env
        self.single_observation_space = _SpaceProxy(obs_shape)
        self.single_action_space = env.single_action_space

    def __getattr__(self, name):
        return getattr(self._env, name)


class _BaseImagePolicyWrapper:
    model_cls = None
    default_channels_last = False

    def __init__(  # noqa: PLR0913
        self,
        env,
        input_size=512,
        hidden_size=512,
        output_size=512,
        framestack=1,
        flat_size=64 * 6 * 9,
        channels_last=None,
        **kwargs,
    ):
        if channels_last is None:
            channels_last = self.default_channels_last

        self._channels_last = channels_last
        self._is_dict_obs = False
        self._native_dtype = None
        self._obs_shape = env.single_observation_space.shape
        self._model_obs_shape = self._obs_shape
        self._view_obs_shape = self._obs_shape

        try:
            raw_obs_space = env.env.observation_space
        except Exception:
            raw_obs_space = getattr(env, "observation_space", None)

        if isinstance(raw_obs_space, pufferlib.spaces.Dict) or hasattr(raw_obs_space, "spaces"):
            self._is_dict_obs = True
            self._native_dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            spaces = raw_obs_space.spaces if hasattr(raw_obs_space, "spaces") else raw_obs_space
            image_space = spaces.get("image")
            if image_space is None:
                for space in spaces.values():
                    if hasattr(space, "shape") and len(space.shape) == IMAGE_DIMS:
                        image_space = space
                        break
            if image_space is not None:
                self._obs_shape = image_space.shape
        elif len(self._obs_shape) == 1 and hasattr(raw_obs_space, "shape"):
            raw_shape = getattr(raw_obs_space, "shape", None)
            if raw_shape is not None and len(raw_shape) == IMAGE_DIMS:
                self._obs_shape = raw_shape

        if len(self._obs_shape) == 1:
            flat_size = int(self._obs_shape[0])
            side = int(flat_size**0.5)
            if side * side != flat_size:
                raise ValueError(
                    "Cannot infer 2D image shape from flat observation size "
                    f"{flat_size}. Please provide an image observation."
                )
            self._view_obs_shape = (side, side, 1)
            self._model_obs_shape = (
                self._view_obs_shape if channels_last else (1, side, side)
            )
        else:
            self._view_obs_shape = self._obs_shape
            if channels_last and self._obs_shape[0] <= 4 and self._obs_shape[-1] > 4:
                c, h, w = self._obs_shape
                self._view_obs_shape = (h, w, c)
                self._model_obs_shape = self._view_obs_shape
            elif not channels_last and self._obs_shape[-1] <= 4 and self._obs_shape[0] > 4:
                h, w, c = self._obs_shape
                self._model_obs_shape = (c, h, w)
            else:
                self._model_obs_shape = self._obs_shape

        env_for_model = _EnvProxy(env, self._model_obs_shape)
        super().__init__(
            env=env_for_model,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
            channels_last=channels_last,
            **kwargs,
        )

    def _prepare_observations(self, observations):
        x = observations
        if self._is_dict_obs:
            x = pufferlib.pytorch.nativize_tensor(observations, self._native_dtype)
            if isinstance(x, dict):
                if "image" in x:
                    x = x["image"]
                else:
                    image_tensor = None
                    for value in x.values():
                        if hasattr(value, "ndim") and value.ndim >= IMAGE_DIMS:
                            image_tensor = value
                            break
                    if image_tensor is None:
                        raise ValueError("Could not find 3D image tensor in dict observation.")
                    x = image_tensor

        if x.ndim == BATCH_FLAT_DIMS:
            # Always unflatten to channel-last first, then normalize layout below.
            x = x.view(x.shape[0], *self._view_obs_shape)

        if x.ndim == IMAGE_DIMS:
            x = x.unsqueeze(0)

        if self._channels_last:
            if x.ndim == IMAGE_DIMS + 1 and x.shape[1] <= 4 and x.shape[-1] > 4:
                x = x.permute(0, 2, 3, 1)
        elif x.ndim == IMAGE_DIMS + 1 and x.shape[-1] <= 4 and x.shape[1] > 4:
            x = x.permute(0, 3, 1, 2)

        return x.contiguous()

    def encode_observations(self, observations, state=None):
        x = self._prepare_observations(observations)
        return super().encode_observations(x, state=state)


class Autoencoder(_BaseImagePolicyWrapper, pufferlib.models.Autoencoder):
    pass


class UNet(_BaseImagePolicyWrapper, pufferlib.models.UNet):
    pass
