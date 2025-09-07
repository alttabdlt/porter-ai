#!/bin/bash

# Install script for VLM support
echo "===================="
echo "Installing VLM Support"
echo "===================="

# Activate venv if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install mlx-vlm
echo "Installing mlx-vlm for vision-language models..."
pip install mlx-vlm

echo ""
echo "✅ Installation complete!"
echo ""
echo "Now you can run with real VLM:"
echo "  python start.py --use-vlm --memory-limit 2000"
echo ""
echo "Note: VLM models use 1.5-2GB memory, so increase memory limit"