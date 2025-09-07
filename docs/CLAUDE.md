# CLAUDE.md - Porter.AI / FASTVLM Jarvis Master Reference

## 🎯 Project Overview

**Current State**: Porter.AI - Basic screen monitoring with Apple FastVLM-0.5B
**Target State**: FASTVLM Jarvis - Intelligent, context-aware assistant that continuously watches and helps
**Location**: `/Users/axel/Desktop/Coding-Projects/porter.ai`

---

## 📚 Documentation Structure

### Core Documentation Files

1. **TASKS2.md** - Complete Jarvis Vision & Requirements
   - Product principles (assistive, privacy-first, non-intrusive)
   - Target architecture (continuous capture → context → interventions)
   - Core components specification
   - Implementation phases (10 weeks)
   - Success metrics

2. **MIGRATION_PLAN.md** - Detailed Implementation Roadmap
   - Current vs target gap analysis
   - File-by-file modification guide
   - 9 new components to create
   - 70+ micro-action checklist
   - Week-by-week implementation plan
   - Risk mitigation strategies

---

## 🏗️ Current Implementation

### Working Components (7 Python files)
```
porter.ai/
├── real_screen_capture.py   # Event-driven capture (needs → continuous)
├── real_assistant.py        # Basic loop (needs → context/policy)
├── vlm_processor.py         # FastVLM-0.5B integration (needs → ROI/batching)
├── run_real.py             # Entry point (needs → new components)
├── simple_server.py        # WebSocket server (mostly ready)
├── simple_dashboard.html   # Basic UI (needs → toast/palette)
├── test_full_system.py    # Tests (needs → expansion)
└── quick_test.py          # Quick validation
```

### Current Capabilities
- ✅ Screen capture on change (SSIM detection)
- ✅ Apple FastVLM-0.5B descriptions
- ✅ Basic importance scoring
- ✅ WebSocket dashboard updates
- ✅ Screenshot storage

### Missing Critical Components
- ❌ Continuous capture (15-30 fps)
- ❌ Ring buffer for history
- ❌ OCR text extraction
- ❌ Context tracking
- ❌ Memory system
- ❌ Policy engine
- ❌ Tool router & skills
- ❌ Toast notifications
- ❌ TTS output
- ❌ Privacy controls

---

## 🚀 Quick Start Commands

### Run Current System
```bash
# Basic mode (simplified VLM)
python run_real.py

# Full FastVLM mode
python run_real.py --full-model

# Custom settings
python run_real.py --full-model --threshold 0.3 --interval 2.0
```

### Testing
```bash
# Quick component test
python quick_test.py

# Full system test
python test_full_system.py

# Test VLM specifically
python vlm_processor.py
```

---

## 📋 Implementation Priority

### Week 1: Foundation (FROM MIGRATION_PLAN.md)
1. Fork `real_screen_capture.py` → `continuous_capture.py`
2. Implement ring buffer with deque
3. Add continuous capture thread (15 fps)
4. Create frame windowing methods
5. Add motion detection for saliency

### Critical Path
1. **Continuous Capture** - Everything depends on this
2. **OCR Engine** - Needed for context understanding
3. **Context Fusion** - Core intelligence
4. **Policy Engine** - Smart interventions
5. **Memory System** - Learning and persistence

---

## 🔧 Key Technical Details

### Model
- **Primary**: `apple/FastVLM-0.5B-fp16`
- **Fallback 1**: `InsightKeeper/FastVLM-0.5B-MLX-6bit`
- **Fallback 2**: `mlx-community/Qwen2-VL-2B-Instruct-4bit`

### Performance Targets
- Latency: <300ms end-to-end
- FPS: 15-30 continuous capture
- Memory: <3GB peak usage
- GPU: <10% duty cycle normal use

### Architecture Evolution
```
Current: Event → Capture → Describe → Dashboard
Target:  Continuous → Buffer → ROI+OCR → Context → Policy → Intervention
```

---

## 🎮 When Starting New Chat

1. **Check current state**: Look at which week of MIGRATION_PLAN.md we're on
2. **Reference specs**: Use TASKS2.md for requirements
3. **Follow checklist**: Use micro-actions from MIGRATION_PLAN.md
4. **Test incrementally**: Each component should work standalone

---

## 📝 Important Context

### Design Principles (from TASKS2.md)
- **Assistive, not intrusive**: Only intervene when truly helpful
- **Privacy-first**: Everything on-device, ephemeral buffer
- **Deterministic UX**: Max 1 nudge per N minutes
- **Explainable**: Show ROI + rationale for every suggestion
- **Graceful degradation**: Fall back when GPU busy

### Current Limitations
- Event-driven capture (not continuous)
- No memory/context persistence
- Simple threshold-based decisions
- Dashboard-only output (no toast/TTS)
- No OCR or UI tree parsing

---

## 🔄 Migration Strategy

### Phase 1: Parallel Development
Keep Porter.AI running while building new components

### Phase 2: Integration
Create `run_jarvis.py` as new entry point

### Phase 3: Cutover
Replace old system with Jarvis

---

## 🚨 Critical Files to Never Delete

1. `vlm_processor.py` - Core FastVLM integration
2. `real_screen_capture.py` - Base for continuous capture
3. `simple_dashboard.html` - UI foundation
4. `screenshots/` - Testing data

---

## 📌 Next Immediate Steps

1. Start Week 1 tasks from MIGRATION_PLAN.md
2. Create `continuous_capture.py` based on `real_screen_capture.py`
3. Implement ring buffer for frame history
4. Test 15 fps capture rate
5. Add ROI detection

---

## 🤖 For Claude

When you read this file in a new chat:
1. You're building FASTVLM Jarvis on top of Porter.AI
2. Reference TASKS2.md for what to build
3. Reference MIGRATION_PLAN.md for how to build it
4. Current week: Week 1 (Foundation) 
5. Main focus: Transform event-driven → continuous capture