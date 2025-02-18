#!/bin/bash

MODEL_DIR="/opt/app/models/llama3-8b-instruct"

# Only download if the directory does NOT exist
if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory $MODEL_DIR not found. Downloading..."
  huggingface-cli download meta-llama/Meta-Llama-3-8B \
    --local-dir "$MODEL_DIR" \
    --token "$HUGGINGFACE_HUB_TOKEN"
else
  echo "Model directory $MODEL_DIR already exists. Skipping download..."
fi

exec "$@"