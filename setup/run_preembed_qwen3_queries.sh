#!/bin/bash

set -e

# Expect a GPU id so we can pin the CUDA device for the Python process.
# Keep the CLI minimal: --gpu <device_id>
GPU_DEVICE=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpu)
        GPU_DEVICE="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

if [ -z "$GPU_DEVICE" ]; then
    echo "Error: --gpu argument is required."
    echo "Usage: $0 --gpu <device_id>"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT_PATH="$PROJECT_ROOT/setup/preembed_qwen3_queries.py"

# Read all possible instruction keys from the Python script
INSTRUCTIONS=$(
    sed -n '/^INSTRUCTIONS = {/,/}/p' "$PYTHON_SCRIPT_PATH" |
    grep -E "^\s*['\"][^'\"]+['\"]:" |
    sed -E "s/^\s*['\"]([^'\"]+)['\"].*/\1/"
)

cd "$PROJECT_ROOT"

for instruction in $INSTRUCTIONS; do
    echo "=================================================="
    echo "Running pre-embedding for instruction: $instruction on GPU: $GPU_DEVICE"
    echo "=================================================="
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 setup/preembed_qwen3_queries.py --instruction "$instruction"
    echo ""
done

echo "All pre-embedding tasks are complete."