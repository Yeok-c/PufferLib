#!/bin/bash
# Usage: ./build.sh [local|fast]
# Finds the newest trained weights, exports them, and builds the standalone demo binary.

set -e

PUFFERLIB_DIR="$(pwd)"
EXPERIMENTS_DIR="$PUFFERLIB_DIR/experiments"
RESOURCES_DIR="$PUFFERLIB_DIR/pufferlib/resources/racer_simple"

MODE=${1:-local}
PLATFORM="$(uname -s)"
RAYLIB_NAME='raylib-5.5_macos'
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
fi

# --- Find newest trained weights ---
NEWEST_PT=$(ls -t "$EXPERIMENTS_DIR"/puffer_racer_simple_*/model_*.pt 2>/dev/null | head -1)

if [ -z "$NEWEST_PT" ]; then
    echo "No saved model weights found in $EXPERIMENTS_DIR/puffer_racer_simple_*/"
    echo "Train first with: puffer train racer_simple"
    exit 1
fi

echo "Found trained weights: $NEWEST_PT"

# --- Export weights to .bin ---
echo "Exporting weights..."
source "$PUFFERLIB_DIR/.venv/bin/activate" 2>/dev/null || true
python -c "
import torch, numpy as np
state = torch.load('$NEWEST_PT', map_location='cpu', weights_only=True)
# LSTMWrapper saves lstm.* and cell.* as duplicate tensors; skip cell.*
weights = np.concatenate([v.numpy().flatten() for k, v in state.items() if not k.startswith('cell.')])
weights.tofile('$RESOURCES_DIR/puffer_racer_simple_weights.bin')
print(f'  Exported {len(weights)} weights ({len(weights)*4} bytes)')
"

# --- Build binary ---
FLAGS=(
    -Wall
    -DPLATFORM_DESKTOP
    -DRACER_SIMPLE_DEMO
    -I./$RAYLIB_NAME/include
    -I./pufferlib/extensions
    pufferlib/ocean/racer_simple/racer_simple.c -o racer_simple
    ./$RAYLIB_NAME/lib/libraylib.a
    -lm -lpthread
    -ferror-limit=3
)

if [ "$PLATFORM" = "Darwin" ]; then
    FLAGS+=(-framework Cocoa -framework IOKit -framework CoreVideo)
fi

if [ "$MODE" = "local" ]; then
    echo "Building racer_simple (debug)..."
    clang -g -O0 "${FLAGS[@]}"
elif [ "$MODE" = "fast" ]; then
    echo "Building racer_simple (optimized)..."
    clang -O2 -DNDEBUG "${FLAGS[@]}"
else
    echo "Usage: ./build.sh [local|fast]"
    exit 1
fi

echo ""
echo "Done. To launch the demo, run:"
echo "  ./racer_simple"
