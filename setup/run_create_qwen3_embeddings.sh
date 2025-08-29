#!/bin/bash

# This script runs the embedding creation script for Qwen3 with all possible instructions.
# It should be run from the root of the project directory.

set -e

GPU_DEVICE=""
INSTRUCTION_INDEX=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpu)
        GPU_DEVICE="$2"
        shift # past argument
        shift # past value
        ;;
        --instruction)
        INSTRUCTION_INDEX="$2"
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
    echo "Usage: $0 --gpu <device_id> [--instruction <index>]"
    echo "  --gpu <device_id>: GPU device to use (required)"
    echo "  --instruction <index>: Select specific instruction by index (0-based, optional)"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT_PATH="$PROJECT_ROOT/setup/create_qwen3_embeddings.py"

# Using all possible instructions (by iterating keys in file)
INSTRUCTIONS=$(
    sed -n '/^EMBED_INSTRUCTS = {/,/}/p' "$PYTHON_SCRIPT_PATH" |
    grep -E "^\s*['\"][^'\"]+['\"]:" |
    sed -E "s/^\s*['\"]([^'\"]+)['\"].*/\1/"
)

cd "$PROJECT_ROOT"

if [ -n "$INSTRUCTION_INDEX" ]; then
    INSTRUCTIONS_ARRAY=($INSTRUCTIONS)
    
    if [[ "$INSTRUCTION_INDEX" =~ ^[0-9]+$ ]] && [ "$INSTRUCTION_INDEX" -ge 0 ] && [ "$INSTRUCTION_INDEX" -lt "${#INSTRUCTIONS_ARRAY[@]}" ]; then
        SELECTED_INSTRUCTION="${INSTRUCTIONS_ARRAY[$INSTRUCTION_INDEX]}"
        echo "=================================================="
        echo "Running embedding creation for instruction #$INSTRUCTION_INDEX: $SELECTED_INSTRUCTION on GPU: $GPU_DEVICE"
        echo "=================================================="
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 setup/create_qwen3_embeddings.py --instruct "$SELECTED_INSTRUCTION"
        echo ""
        echo "Embedding creation for instruction '$SELECTED_INSTRUCTION' is complete."
    else
        echo "Error: Invalid instruction index. Must be between 0 and $((${#INSTRUCTIONS_ARRAY[@]} - 1))"
        echo "Available instructions:"
        for i in "${!INSTRUCTIONS_ARRAY[@]}"; do
            echo "  $i: ${INSTRUCTIONS_ARRAY[$i]}"
        done
        exit 1
    fi
else
    for instruction in $INSTRUCTIONS; do
        echo "=================================================="
        echo "Running embedding creation for instruction: $instruction on GPU: $GPU_DEVICE"
        echo "=================================================="
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 setup/create_qwen3_embeddings.py --instruct "$instruction"
        echo ""
    done
    echo "All embedding creation tasks are complete."
fi