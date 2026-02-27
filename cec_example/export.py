"""Export trained DASM policy weights to a flat binary file.

The binary file contains all parameter tensors concatenated in
parameter-iteration order (matching model.named_parameters()).
This is the format expected by C inference code.

Usage:
    python src/export.py --checkpoint experiments/dasm_.../model_dasm_000250.pt
    python src/export.py --checkpoint latest
    python src/export.py --checkpoint path/to/model.pt --output weights.bin
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

import pufferlib
import pufferlib.vector
from pufferlib.ocean.dasm.dasm import Dasm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy import Policy


def parse_args():
    p = argparse.ArgumentParser(description="Export DASM policy weights")
    p.add_argument("--checkpoint", type=str, required=True,
                    help='Path to .pt file, or "latest"')
    p.add_argument("--output", type=str, default="dasm_weights.bin")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument("--num-blocks", type=int, default=5)
    p.add_argument("--data-dir", type=str, default="experiments")
    return p.parse_args()


def main():
    args = parse_args()

    env = Dasm(num_envs=1, num_agents=args.num_agents, num_blocks=args.num_blocks)
    policy = Policy(env, hidden_size=args.hidden_size).to(args.device)

    checkpoint = args.checkpoint
    if checkpoint == "latest":
        candidates = glob.glob(os.path.join(args.data_dir, "dasm*/*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoints in " + args.data_dir)
        checkpoint = max(candidates, key=os.path.getctime)

    state_dict = torch.load(checkpoint, map_location=args.device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    print(f"Loaded: {checkpoint}")

    weights = []
    for name, param in policy.named_parameters():
        flat = param.data.cpu().numpy().flatten()
        weights.append(flat)
        print(f"  {name:40s} {str(list(param.shape)):20s}  first={flat[0]:.6f}")

    weights = np.concatenate(weights)
    weights.tofile(args.output)
    print(f"\nExported {len(weights):,} weights to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
