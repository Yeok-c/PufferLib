#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Example weights live at: cec_example/puffer_racer_simple_example_weights.pt
DEFAULT_DEMO_WEIGHTS="${REPO_ROOT}/cec_example/puffer_racer_simple_example_weights.pt"

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  $0 eval --load-model-path <path>"
  echo "  $0 demo [--load-model-path <path>]"
  echo "  $0 train"
  exit 1
fi

cmd="$1"
shift || true

case "${cmd}" in
  eval)
    # Expect: launch.sh eval --load-model-path <path>
    if [[ $# -lt 2 ]] || [[ "$1" != "--load-model-path" ]]; then
      echo "Usage: $0 eval --load-model-path <path>"
      exit 1
    fi

    LOAD_MODEL_PATH="$2"
    echo "Evaluating model ${LOAD_MODEL_PATH} and saving gif to ./cec_example/eval.gif"
    puffer eval puffer_racer_simple \
      --save-frames 2000 \
      --gif-path ./cec_example/eval.gif \
      --load-model-path "${LOAD_MODEL_PATH}"
    ;;

  demo)
    # Optional: --load-model-path <path>, default to example weights
    LOAD_MODEL_PATH="${DEFAULT_DEMO_WEIGHTS}"
    if [[ $# -ge 2 ]] && [[ "$1" == "--load-model-path" ]]; then
      LOAD_MODEL_PATH="$2"
    fi

    # Build and run the racer_simple demo from the repo root,
    # since build.sh expects to be run there and outputs ./racer_simple.
    (
      cd "${REPO_ROOT}"
      bash ./pufferlib/ocean/racer_simple/build.sh
      ./racer_simple --load-model-path "${LOAD_MODEL_PATH}"
    )
    ;;

  train)
    # If the user supplies extra args, pass them through to puffer
    puffer train puffer_racer_simple "$@"
    ;;

  *)
    echo "Unknown command: ${cmd}"
    echo "Valid commands are: eval, demo, train"
    exit 1
    ;;
esac

