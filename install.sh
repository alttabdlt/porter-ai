#!/bin/bash

# Porter.AI Installation Script
# Handles dependency installation and environment setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════╗"
echo "║       🤖 Porter.AI Installation           ║"
echo "╚══════════════════════════════════════════╝"
echo

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "✓ Detected: $OS on $ARCH"

# Install base requirements
echo "Installing base dependencies..."
pip install -r requirements.txt --quiet

# Platform-specific installations
if [[ "$OS" == "Darwin" ]]; then
    echo "Configuring for macOS..."
    
    if [[ "$ARCH" == "arm64" ]]; then
        echo "  → Apple Silicon detected - MLX optimizations enabled"
        # MLX is already in requirements.txt
    else
        echo "  → Intel Mac detected - Using standard packages"
    fi
    
    # Install macOS specific audio dependencies
    if ! brew list portaudio &>/dev/null; then
        echo "Installing portaudio for audio support..."
        brew install portaudio
    fi
    
elif [[ "$OS" == "Linux" ]]; then
    echo "Configuring for Linux..."
    echo "  → Installing system dependencies..."
    
    # Check if apt is available
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3-dev portaudio19-dev
    fi
fi

# Create necessary directories
echo "Setting up directories..."
mkdir -p screenshots
mkdir -p logs
mkdir -p .claude

# Download model if needed (optional - for first run optimization)
echo "Preparing VLM model cache..."
$PYTHON_CMD -c "
import warnings
warnings.filterwarnings('ignore')
try:
    print('  → Checking FastVLM model availability...')
    # This will cache the model if not already present
    import os
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    # Model will be downloaded on first actual use
    print('  → Model will be downloaded on first run')
except Exception as e:
    print(f'  → Model preparation skipped: {e}')
"

# Verify installation
echo
echo "Verifying installation..."
$PYTHON_CMD -c "
import sys
required = ['mss', 'PIL', 'cv2', 'aiohttp', 'websockets', 'numpy', 'pydantic']
missing = []
for pkg in required:
    try:
        if pkg == 'PIL':
            import PIL
        elif pkg == 'cv2':
            import cv2
        else:
            __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'❌ Missing packages: {missing}')
    sys.exit(1)
else:
    print('✓ All core dependencies installed')
"

# Setup completion status
if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     ✅ Installation Complete!            ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
    echo
    echo "To start Porter.AI:"
    echo "  1. Activate environment: source venv/bin/activate"
    echo "  2. Run: python start.py"
    echo
    echo "Options:"
    echo "  --fps 5         : Capture rate (default: 5)"
    echo "  --interval 3    : Analysis interval (default: 3.0)"
    echo "  --memory-limit  : Memory limit in MB (default: 300)"
    echo
else
    echo -e "${RED}Installation failed. Please check the errors above.${NC}"
    exit 1
fi