import sys
import gymnasium as gym

import pufferlib
import pufferlib.utils

def test_suppress():
    with pufferlib.utils.Suppress():
        # Use a core env to avoid optional Atari dependencies/ROMs
        gym.make('CartPole-v1')
        print('stdout (you should not see this)', file=sys.stdout)
        print('stderr (you should not see this)', file=sys.stderr)

if __name__ == '__main__':
    test_suppress()