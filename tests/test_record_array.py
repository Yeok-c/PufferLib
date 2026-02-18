import gymnasium as gym
import numpy as np


def create_dtype_from_space(space) -> np.dtype:
    """Create a structured dtype that matches a Gymnasium space."""
    if isinstance(space, gym.spaces.Dict):
        dtype_fields = [(name, create_dtype_from_space(subspace)) for name, subspace in space.spaces.items()]
        return np.dtype(dtype_fields)
    if isinstance(space, gym.spaces.Tuple):
        dtype_fields = [(f"field{i}", create_dtype_from_space(subspace)) for i, subspace in enumerate(space.spaces)]
        return np.dtype(dtype_fields)
    if isinstance(space, gym.spaces.Box):
        # Represent Box as a fixed-shape array field
        return np.dtype((space.dtype, space.shape))
    if isinstance(space, gym.spaces.Discrete):
        return np.dtype(np.int64)
    raise TypeError(f"Unsupported space type: {type(space)}")


def test_record_array_roundtrip():
    space = gym.spaces.Dict(
        {
            "position": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "velocity": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        }
    )

    dtype = create_dtype_from_space(space)
    sample = space.sample()

    arr = np.zeros((), dtype=dtype)
    arr["position"] = sample["position"]
    arr["velocity"] = sample["velocity"]

    # Basic sanity checks: fields present and values preserved
    assert np.allclose(arr["position"], sample["position"])
    assert np.allclose(arr["velocity"], sample["velocity"])
