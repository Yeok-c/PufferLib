#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Example weights live at: cec_example/puffer_racer_simple_example_weights.pt
DEFAULT_DEMO_WEIGHTS="${REPO_ROOT}/cec_example/puffer_racer_simple_example_weights.pt"
EXPERIMENTS_DIR="${REPO_ROOT}/experiments"

# Resolve model path: explicit path, else latest in experiments/, else example weights
resolve_load_model_path() {
  if [[ $# -ge 2 ]] && [[ "$1" == "--load-model-path" ]]; then
    echo "$2"
    return
  fi
  local latest
  latest=$(ls -t "${EXPERIMENTS_DIR}"/puffer_racer_simple*.pt 2>/dev/null | head -1)
  if [[ -n "${latest}" ]]; then
    echo "${latest}"
  else
    echo "${DEFAULT_DEMO_WEIGHTS}"
  fi
}

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  $0 eval [--load-model-path <path>]   # default: latest in experiments/ or example weights"
  echo "  $0 demo [--load-model-path <path>]   # default: latest in experiments/ or example weights"
  echo "  $0 train"
  exit 1
fi

cmd="$1"
shift || true

case "${cmd}" in
  eval)
    LOAD_MODEL_PATH=$(resolve_load_model_path "$@")
    echo "Evaluating model ${LOAD_MODEL_PATH} and saving gif to ./eval.gif"
    puffer eval puffer_racer_simple \
      --save-frames 2000 \
      --gif-path ./eval.gif \
      --load-model-path "${LOAD_MODEL_PATH}"
    ;;

  demo)
    LOAD_MODEL_PATH=$(resolve_load_model_path "$@")

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

