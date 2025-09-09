# 🚀 Porter.AI Cross-Platform Setup Guide

## Overview

Porter.AI now supports **Windows, macOS, and Linux** with automatic platform detection and optimized backends for each system.

## 🏗️ Architecture

```
Porter.AI
├── Platform Detection (automatic)
├── VLM Backend Selection
│   ├── OmniVLM (968M) - Cross-platform default
│   ├── FastVLM (500M) - Apple Silicon only
│   └── Simple Processor - Fallback/testing
└── Screen Capture
    ├── Cross-platform (mss) - Works everywhere
    └── Native macOS (ScreenCaptureKit) - Optional
```

## 🖥️ Platform Support

| Platform | Screen Capture | VLM Models | Performance |
|----------|---------------|------------|-------------|
| **Windows 11** | ✅ mss | ✅ OmniVLM, Moondream | 2-3s/frame (CPU) |
| **macOS (Apple Silicon)** | ✅ mss/native | ✅ All models + FastVLM | <1s/frame (MLX) |
| **Linux/Codespaces** | ⚠️ mss (no display) | ✅ OmniVLM (CPU only) | 3-5s/frame |

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/porter-ai.git
cd porter-ai
```

### 2. Install Platform-Specific Requirements

#### Windows
```bash
pip install -r requirements-windows.txt
```

#### macOS
```bash
pip install -r requirements-macos.txt
```

#### Linux/Codespaces
```bash
pip install -r requirements-linux.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your preferences
```

Key settings in `.env`:
- `USE_OMNIVLM=true` - Use cross-platform OmniVLM
- `USE_FASTVLM=false` - Use Apple FastVLM (Mac only)
- `CAPTURE_FPS=10` - Frames per second
- `OMNIVLM_DEVICE=auto` - auto/cpu/cuda/mps

## 🧪 Testing

### Quick Test
```bash
python test_cross_platform.py
```

This will test:
- Platform detection
- Screen capture (if display available)
- VLM processor
- Integration pipeline

### Run Main Application
```bash
python app/streaming/main_streaming.py
```

Then open: http://localhost:8000/streaming

## 💻 Development Workflow

### On GitHub Codespaces
1. **Develop** in Codespaces (no GPU, no display)
2. **Test** with mock data and SimplifiedVLMProcessor
3. **Commit** changes

```bash
# In Codespaces
pip install -r requirements-base.txt
python test_cross_platform.py  # Tests with mock data
```

### On Your Local Machine (Windows/Mac)
1. **Pull** latest changes
2. **Test** with real screen capture and VLM
3. **Verify** visual output

```bash
# On Windows laptop
git pull origin main
pip install -r requirements-windows.txt
python app/streaming/main_streaming.py
# Open http://localhost:8000/streaming
```

## 🚀 Performance Optimization

### Windows
- ONNX Runtime provides 2-4x speedup
- Use `OMNIVLM_DEVICE=cuda` if NVIDIA GPU available
- Enable DirectML for Windows ML acceleration

### macOS (Apple Silicon)
- Use `USE_FASTVLM=true` for 85x faster inference
- MPS acceleration with `OMNIVLM_DEVICE=mps`
- Native ScreenCaptureKit for better performance

### Linux
- Use `OMNIVLM_USE_FP16=true` for faster inference
- Install CUDA toolkit for GPU support
- Use Xvfb for headless operation

## 🔧 Troubleshooting

### "No display available" (Linux/Codespaces)
```bash
# Install virtual display
sudo apt-get install xvfb
# Run with virtual display
xvfb-run -a python app/streaming/main_streaming.py
```

### "Model download slow"
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('NexaAI/OmniVLM-968M')"
```

### "Out of memory"
- Set `USE_QUANTIZATION=true` in .env
- Use `OMNIVLM_USE_FP16=true`
- Reduce `CAPTURE_WIDTH` and `CAPTURE_HEIGHT`

## 📊 Model Comparison

| Model | Size | Speed | Memory | Platforms |
|-------|------|-------|---------|-----------|
| **OmniVLM** | 968M | 2-3s/frame | ~1GB | All |
| **FastVLM** | 500M | <1s/frame | <2GB | macOS only |
| **Moondream** | 500M-2B | 1-2s/frame | 1-2GB | All |
| **Simple** | 0 | Instant | <100MB | All |

## 🔐 Privacy & Security

- All processing happens locally
- No data sent to external servers
- Screen capture requires explicit permissions
- Set `LOCAL_ONLY=true` to restrict network access

## 📝 Next Steps

1. **Install dependencies** for your platform
2. **Configure .env** with your preferences
3. **Run tests** to verify setup
4. **Start developing** with cross-platform support!

## 🤝 Contributing

- Develop on any platform
- Test on your target platforms
- Submit PRs with platform-specific notes

## 📚 Resources

- [OmniVLM Documentation](https://huggingface.co/NexaAI/OmniVLM-968M)
- [FastVLM Paper](https://arxiv.org/abs/2412.13303)
- [mss Documentation](https://python-mss.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)