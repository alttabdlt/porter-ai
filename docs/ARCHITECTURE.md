# Porter.AI Architecture Documentation

## System Overview

Porter.AI is a real-time screen intelligence system that leverages Apple's FastVLM-0.5B model to provide continuous understanding of user activities. The system operates at 16 FPS on Apple Silicon with minimal memory footprint.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User's Screen Activity                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ScreenCaptureKit                          │
│                   (30 FPS Native Capture)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Adaptive Frame Sampler                      │
│              (Intelligent 10 FPS Selection)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastVLM-0.5B                             │
│          (Vision-Language Model Processing)                  │
│                                                              │
│  • FastViTHD Vision Encoder (3.4x smaller)                  │
│  • Qwen2-0.5B Language Model                                │
│  • MLX Framework (Apple Silicon Optimized)                  │
│  • 8-bit Quantization                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Context Fusion                            │
│              (Simplified Aggregation Layer)                  │
│                                                              │
│  • Temporal Smoothing                                       │
│  • Activity Detection                                       │
│  • Entity Extraction                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   WebSocket Server                           │
│                    (Port 8001)                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard                             │
│                 (http://localhost:8000)                      │
│                                                              │
│  • Real-time Screen Preview                                 │
│  • AI Descriptions                                          │
│  • Performance Metrics                                      │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Screen Capture Layer

**ScreenCaptureKit Integration** (`simple_screencapture.py`)
- Native macOS API for efficient screen capture
- 30 FPS source capture rate
- Hardware-accelerated encoding
- Minimal CPU overhead

**Frame Buffer Management**
- Queue size: 3 frames (reduced from 10)
- Non-blocking frame drops when queue full
- Automatic memory pressure handling

### 2. Adaptive Frame Sampling

**Intelligent Selection** (`screencapture_kit.py`)
- Base sampling rate: 10 FPS
- Activity-based adaptive sampling
- Motion detection for dynamic adjustment
- Reduces VLM processing load

### 3. FastVLM Processing

**Model Architecture**
```
Input Image (1920x1080)
    ↓
FastViTHD Vision Encoder
    ↓
Vision Features (compressed)
    ↓
Qwen2-0.5B Language Model
    ↓
Text Description (150 tokens max)
```

**Key Optimizations**
- 8-bit quantization for memory efficiency
- MLX framework for Apple Silicon acceleration
- Context window: Previous frame description
- Temperature: 0.5 (optimized for consistency)

**Performance Metrics**
- Inference time: ~60ms per frame
- Memory usage: <1GB for model
- TTFT: 85x faster than LLaVA-OneVision

### 4. Context Fusion

**Simplified Processing** (`context_fusion.py`)
```python
FusedContext:
├── timestamp: float
├── description: str (from VLM)
├── confidence: float (0.8 fixed)
├── importance: float (0.7 default)
├── application: str (extracted)
├── activity_type: str (detected)
├── key_entities: List[str]
└── sources_used: ['vlm']
```

**Temporal Features**
- History buffer: 30 frames
- Activity smoothing over 3 frames
- Entity deduplication
- Context persistence

### 5. Communication Layer

**WebSocket Server** (`server.py`)
- Async WebSocket on port 8001
- HTTP server on port 8000
- Message types:
  - `frame`: Base64 encoded JPEG
  - `context`: AI analysis results
  - `metrics`: Performance data

**Data Flow**
```json
{
  "type": "context",
  "data": {
    "description": "User editing video in Final Cut Pro",
    "confidence": 0.8,
    "activity": "editing",
    "timestamp": 1234567890.123
  }
}
```

### 6. Web Dashboard

**Frontend Components** (`streaming.html`)
- Real-time canvas rendering
- WebSocket event handling
- Performance monitoring
- Activity timeline

## Memory Management

### Allocation Strategy
```
Total Memory Budget: ~2GB
├── FastVLM Model: 900MB
├── Frame Buffers: 300MB (3 frames)
├── Context History: 50MB
├── WebSocket Buffers: 100MB
└── System Overhead: 650MB
```

### Optimization Techniques
1. **Frame Queue Limiting**: Max 3 frames in pipeline
2. **Lazy Loading**: Model loaded only when needed
3. **Garbage Collection**: Forced GC after heavy processing
4. **Buffer Recycling**: Reuse frame buffers

## Data Flow

### Frame Processing Pipeline
```
1. Screen Capture (30 FPS)
   ↓
2. Frame Sampler (10 FPS effective)
   ↓
3. Frame Queue (max 3)
   ↓
4. VLM Processing (16 FPS output)
   ↓
5. Context Generation
   ↓
6. WebSocket Broadcast
   ↓
7. Dashboard Update
```

### Latency Breakdown
- Capture: 5ms
- Sampling: 1ms
- Queue: 0-30ms
- VLM: 60ms
- Context: 2ms
- Network: 5ms
- **Total: ~75-105ms end-to-end**

## Configuration

### Environment Variables
```bash
# Model path
FASTVLM_MODEL_PATH=ml-fastvlm/models/fastvlm-0.5b-mlx

# Performance tuning
MAX_FRAMES_IN_QUEUE=3
BASE_SAMPLING_FPS=10
VLM_MAX_TOKENS=150
VLM_TEMPERATURE=0.5
```

### Runtime Parameters
```python
StreamingPipeline({
    'width': 1920,
    'height': 1080,
    'fps': 10,
    'show_cursor': True,
    'use_vlm': True
})
```

## Scaling Considerations

### Horizontal Scaling Options
1. **Multi-Model Pipeline**: Run multiple VLM instances
2. **Distributed Processing**: Split screen regions
3. **Cloud Hybrid**: Local preview + cloud analysis

### Vertical Scaling Path
```
Local Device Tiers:
├── FastVLM-0.5B @ 16 FPS (current)
├── FastVLM-1.5B @ 8-10 FPS
└── FastVLM-7B @ 2-4 FPS

Cloud API Tiers:
├── Gemini 1.5 Flash (cost-effective)
├── Claude 3 Sonnet (balanced)
└── GPT-4V (premium)
```

## Security Considerations

### Privacy Features
- All processing local by default
- No data leaves device without consent
- Screen recording permissions required
- Secure WebSocket connections

### Data Protection
- Frames encrypted in transit
- No persistent storage of captures
- Memory cleared on shutdown
- User-controlled cloud options

## Future Architecture Enhancements

### Planned Improvements
1. **Plugin System**: Extensible processors
2. **Multi-Display**: Support multiple screens
3. **Activity Recording**: Save sessions with annotations
4. **Natural Language Interface**: Query past activities
5. **Custom Model Training**: Fine-tune for specific use cases

### Experimental Features
- WebGPU acceleration
- Streaming to mobile devices
- Collaborative screen sharing
- Real-time translation of UI

## Monitoring & Debugging

### Key Metrics
```python
{
    'fps': float,           # Current frame rate
    'latency': float,       # End-to-end latency (ms)
    'cpu': float,           # CPU usage (%)
    'memory': float,        # Memory usage (MB)
    'frames_processed': int,
    'frames_analyzed': int,
    'model_inference_time': float
}
```

### Debug Endpoints
- `/metrics`: Real-time performance data
- `/status`: System health check
- `/config`: Current configuration
- `/logs`: Recent log entries

## Deployment

### Development
```bash
python app/streaming/main_streaming.py
```

### Production
```bash
# With process manager
pm2 start app/streaming/main_streaming.py --name porter-ai

# With systemd service
systemctl start porter-ai
```

### Docker (Future)
```dockerfile
FROM python:3.12-slim
# MLX requires native Apple Silicon
# Docker support pending MLX compatibility
```

---

*Architecture Version: 2.0.0 | Last Updated: September 2025*