# FastVLM Model Setup

The FastVLM-0.5B model is required for the streaming architecture but is too large for GitHub (1.1GB).

## Download the Model

1. The model should be placed at: `ml-fastvlm/models/fastvlm-0.5b-mlx/`

2. You can obtain it from:
   - The official Apple FastVLM repository
   - Or convert from the checkpoints using the mlx-vlm conversion script

## Model Structure

The model directory should contain:
- `model.safetensors` (642MB)
- `fastvithd.mlpackage/` (468MB)
- Configuration files (config.json, tokenizer files, etc.)

## Alternative

The system will automatically fall back to a simplified processor if the model is not available, though with reduced capabilities.