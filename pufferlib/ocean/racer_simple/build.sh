#!/bin/bash
# Usage: ./build.sh [local|fast]

# 1. Clear the terminal/cell immediately
# \033[H (home cursor) \033[J (clear to end of screen)
printf "\033[H\033[J"

set -e

PUFFERLIB_DIR="$(pwd)"
EXPERIMENTS_DIR="$PUFFERLIB_DIR/experiments"
RESOURCES_DIR="$PUFFERLIB_DIR/pufferlib/resources/racer_simple"
MODE=${1:-local}
PLATFORM="$(uname -s)"
RAYLIB_NAME=$([ "$PLATFORM" = "Linux" ] && echo "raylib-5.5_linux_amd64" || echo "raylib-5.5_macos")

echo "--- PufferLib Build Pipeline ---"

# --- Find newest trained weights ---
NEWEST_PT=$(ls -t "$EXPERIMENTS_DIR"/puffer_racer_simple_*/model_*.pt 2>/dev/null | head -1)

if [ -z "$NEWEST_PT" ]; then
    echo "❌ Error: No weights found in $EXPERIMENTS_DIR"
    echo "   Run: puffer train racer_simple"
    exit 1
fi

echo "📂 Using: $(basename "$NEWEST_PT")"

# --- Export weights to .bin ---
# We use a carriage return (\r) in the python print to keep it on one line
source "$PUFFERLIB_DIR/.venv/bin/activate" 2>/dev/null || true
python -c "
import torch, numpy as np, sys
try:
    state = torch.load('$NEWEST_PT', map_location='cpu', weights_only=True)
    weights = np.concatenate([v.numpy().flatten() for k, v in state.items() if not k.startswith('cell.')])
    weights.tofile('$RESOURCES_DIR/puffer_racer_simple_weights.bin')
    sys.stdout.write(f'\r📦 Exported {len(weights)} weights ({len(weights)*4/1024:.1f} KB) successfully.\n')
except Exception as e:
    print(f'\n❌ Export failed: {e}')
    sys.exit(1)
"

# --- Build binary ---
FLAGS=(
    -Wall -DPLATFORM_DESKTOP -DRACER_SIMPLE_DEMO
    -I./$RAYLIB_NAME/include -I./pufferlib/extensions
    pufferlib/ocean/racer_simple/racer_simple.c -o racer_simple
    ./$RAYLIB_NAME/lib/libraylib.a -lm -lpthread -ferror-limit=3
)
[[ "$PLATFORM" = "Darwin" ]] && FLAGS+=(-framework Cocoa -framework IOKit -framework CoreVideo)

if [ "$MODE" = "local" ]; then
    echo "🛠️  Compiling (Debug/O0)..."
    clang -g -O0 "${FLAGS[@]}"
elif [ "$MODE" = "fast" ]; then
    echo "⚡ Compiling (Optimized/O2)..."
    clang -O2 -DNDEBUG "${FLAGS[@]}"
else
    echo "Usage: ./build.sh [local|fast]"; exit 1
fi

echo -e "\n✅ Build Complete!"
echo "🚀 Run: ./racer_simple"