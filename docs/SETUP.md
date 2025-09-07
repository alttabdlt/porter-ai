# 🚀 Porter.AI Setup Guide

## Prerequisites
- Python 3.8+
- 4GB RAM minimum
- macOS, Linux, or Windows

## Installation

### Quick Install
```bash
# macOS/Linux
chmod +x install.sh && ./install.sh

# Windows
install.bat
```

## Running Porter.AI

```bash
# Start with defaults
python start.py

# Custom settings (optional)
python start.py --fps 5 --interval 3 --memory-limit 300
```

## Dashboard

Access the monitoring dashboard at:
```
http://localhost:8000
```

## Troubleshooting

### Installation Issues
- **macOS**: Run `brew install portaudio` if audio fails
- **Linux**: Run `sudo apt-get install python3-dev portaudio19-dev`
- **Windows**: Download PyAudio wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

### Performance Issues
- Reduce capture rate: `python start.py --fps 2`
- Lower memory limit: `python start.py --memory-limit 200`

### High Memory on Apple Silicon
MLX framework pre-allocates ~440MB - this is normal.

## That's It!

Your intelligent screen monitoring system is ready. Press Ctrl+C to stop.