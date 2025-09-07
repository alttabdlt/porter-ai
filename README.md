# 🤖 Porter.AI - FASTVLM Jarvis

An intelligent screen monitoring assistant with real-time analysis and smart memory management.

## 🚀 Quick Start

```bash
# Run the system
python start.py

# With custom settings
python start.py --fps 5 --interval 3 --memory-limit 300
```

## 📁 Project Structure

```
porter.ai/
├── app/
│   ├── frontend/       # Web dashboard
│   │   └── index.html  # Real-time monitoring UI
│   └── backend/        # Core system
│       ├── main.py     # Main entry point
│       ├── continuous_capture.py  # Screen capture
│       ├── vlm_processor.py      # Vision-Language Model
│       ├── server.py             # WebSocket server
│       └── memory_safety_config.py  # Memory management
├── docs/               # Documentation
│   ├── TASKS2.md      # Project vision
│   ├── MIGRATION_PLAN.md  # Implementation roadmap
│   └── MEMORY_FIX_PLAN.md  # Memory optimization
├── screenshots/        # Captured screenshots
└── start.py           # Startup script
```

## ✨ Features

- **Continuous Screen Monitoring** - 5 FPS capture with motion detection
- **Smart Screenshot Management** - Tier-based retention (Critical/Important/Routine)
- **Memory Safety** - Automatic cleanup, hard limits, crash prevention
- **Real-time Dashboard** - WebSocket updates, memory tracking, TTL timers
- **VLM Analysis** - Screen descriptions and risk assessment

## 📊 Memory Management

The system tracks and manages memory to prevent crashes:
- **Frontend**: 50MB screenshot cache with priority eviction
- **Backend**: 300-500MB limit with automatic cleanup
- **Monitoring**: Real-time memory stats in dashboard

## 🎯 Dashboard Features

Access at `http://localhost:8000`:
- Live screen activity feed
- Tier-based event classification
- Memory usage indicators
- TTL countdown timers
- Automatic screenshot cleanup

## ⚙️ Configuration

Default settings (optimized for stability):
- **FPS**: 5 (capture rate)
- **Buffer**: 5 seconds
- **Analysis Interval**: 3 seconds
- **Memory Limit**: 300MB
- **Screenshot Quality**: 60%

## 🛡️ Safety Features

- Automatic memory limit enforcement
- Graceful degradation under load
- Buffer clearing when critical
- FPS reduction when needed
- Forced garbage collection

## 📝 License

MIT