# Porter.AI - Real-Time Screen Intelligence with FastVLM

Porter.AI is a real-time screen monitoring and intelligence system that uses Apple's FastVLM-0.5B model to provide continuous, contextual understanding of user activities at 16fps on Apple Silicon.

## 🚀 Features

- **Real-time Vision Understanding**: Leverages Apple's FastVLM-0.5B for 85x faster inference than traditional VLMs
- **16 FPS Performance**: Smooth real-time monitoring on MacBook Pro with Apple Silicon
- **Optimized Memory Usage**: Streamlined pipeline with minimal memory footprint (<2GB RAM)
- **Web Dashboard**: Real-time visualization of screen activity and AI insights
- **Privacy-First**: All processing happens locally on your device

## 🏗️ Architecture

### Current Streamlined Pipeline
```
ScreenCaptureKit → FastVLM-0.5B → Context Fusion → WebSocket → Dashboard
```

### Key Components

- **ScreenCaptureKit**: Native macOS screen capture at 30fps source
- **Adaptive Frame Sampler**: Intelligent frame selection for processing
- **FastVLM-0.5B**: Apple's efficient vision-language model (3.4x smaller vision encoder)
- **Context Fusion**: Simplified context aggregation from VLM output
- **WebSocket Server**: Real-time streaming to web dashboard

## 📊 Performance Metrics

- **Model**: FastVLM-0.5B (Apple ML Research)
- **Inference Speed**: ~60ms per frame
- **Frame Rate**: 16 FPS sustained
- **Memory Usage**: <2GB RAM
- **Time-to-First-Token**: 85x faster than LLaVA-OneVision
- **Vision Encoder**: 3.4x smaller than LLaVA-OneVision

## 🛠️ Installation

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.12 (required for coremltools compatibility)
- 8GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/porter.ai.git
cd porter.ai
```

2. Create Python 3.12 virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. FastVLM Model:
The model is already included in `ml-fastvlm/models/fastvlm-0.5b-mlx/`

5. Grant screen recording permissions:
- System Preferences → Privacy & Security → Screen Recording
- Add Terminal/Python to allowed apps

## 🚦 Usage

### Start the streaming pipeline:
```bash
python app/streaming/main_streaming.py
```

### Access the dashboard:
Open your browser and navigate to:
```
http://localhost:8000/streaming
```

The dashboard will show:
- Real-time screen preview
- AI-generated descriptions of your activity
- Performance metrics (FPS, latency, memory usage)
- Activity timeline

## 🔧 Configuration

The system uses optimized defaults, but you can adjust settings in `main_streaming.py`:

```python
pipeline = StreamingPipeline({
    'width': 1920,
    'height': 1080,
    'fps': 10,  # Sampling rate (source is 30fps)
    'show_cursor': True,
    'use_vlm': True  # Enable/disable VLM processing
})
```

## 📁 Project Structure

```
porter.ai/
├── app/
│   ├── streaming/          # Core streaming pipeline
│   │   ├── main_streaming.py     # Main pipeline orchestrator
│   │   ├── simple_screencapture.py  # Screen capture wrapper
│   │   ├── screencapture_kit.py     # Adaptive frame sampling
│   │   └── context_fusion.py        # Context aggregation
│   ├── backend/            # VLM processing & server
│   │   ├── vlm_processor.py  # FastVLM integration
│   │   └── server.py          # WebSocket/HTTP server
│   └── frontend/           # Web dashboard
│       └── streaming.html     # Real-time monitoring UI
├── ml-fastvlm/            # FastVLM model & tools
│   └── models/
│       └── fastvlm-0.5b-mlx/  # Quantized model files
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md   # System design details
│   └── CHANGELOG.md      # Version history
└── README.md
```

## 🧠 Model Information

### FastVLM-0.5B Details
- **Architecture**: FastViTHD vision encoder + Qwen2-0.5B language model
- **Framework**: MLX (optimized for Apple Silicon)
- **Quantization**: 8-bit for efficient memory usage
- **Context**: Enhanced prompting with previous frame context
- **Temperature**: 0.5 for consistent descriptions

### Model Upgrade Path
- **FastVLM-1.5B**: Better accuracy, ~8-10 FPS locally
- **FastVLM-7B**: Best accuracy, requires cloud deployment
- **Hybrid Mode**: Local 0.5B + cloud API for enhanced analysis

## 🔄 Recent Changes (September 2025)

### Major Architecture Simplification
- **Removed Components**:
  - OCR module (VLM handles text recognition)
  - UI Accessibility Bridge (VLM understands UI elements)
  - Saliency Detection (VLM determines importance)
  
### Memory Optimizations
- Reduced frame queues from 10 to 3 frames
- Simplified context fusion to direct VLM output
- Removed redundant processing layers
- Achieved <2GB total memory usage

### Performance Improvements
- Enhanced VLM prompting for better descriptions
- Added context tracking between frames
- Optimized temperature (0.7 → 0.5) for consistency
- Increased max tokens (100 → 150) for detailed descriptions
- Implemented temporal smoothing for activity detection

## 🚀 Future Roadmap

- [ ] Cloud API integration (Gemini/Claude) for enhanced accuracy
- [ ] Multi-model support (1.5B, 7B variants)
- [ ] Electron desktop app packaging
- [ ] Custom model fine-tuning for specific use cases
- [ ] Video recording with AI annotations
- [ ] Natural language queries about screen activity
- [ ] Export activity summaries and reports

## 🐛 Troubleshooting

### Common Issues

1. **Screen Recording Permission Denied**
   - Go to System Preferences → Privacy & Security → Screen Recording
   - Add Python/Terminal to allowed apps
   - Restart the application

2. **Model Loading Error**
   - Ensure you're using Python 3.12 (not 3.13)
   - Check that ml-fastvlm/models/fastvlm-0.5b-mlx/ exists
   - Verify MLX installation: `pip install mlx mlx-vlm`

3. **Low FPS**
   - Check Activity Monitor for CPU/GPU usage
   - Reduce resolution in configuration
   - Ensure no other heavy processes are running

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Apple ML Research for FastVLM
- MLX team for the optimization framework
- ScreenCaptureKit for efficient screen capture on macOS
- The open-source community for mlx-vlm

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

Built with ❤️ using Apple's FastVLM and MLX framework