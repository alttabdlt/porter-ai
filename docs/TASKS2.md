# FASTVLM Jarvis – Real-Time On-Device Assistant

A pragmatic, privacy-preserving, on-device "mini-Jarvis" that watches the screen continuously, understands context, and surfaces useful, non-annoying interventions. Built on top of existing Porter.AI foundation.

---

## Product Principles

- **Assistive, not intrusive**: Default silent; only intervene when confidence + utility > threshold
- **Privacy-first**: Screen pixels never leave device; ephemeral ring buffer; redaction before storage
- **Deterministic UX cadence**: Maximum 1 nudge per N minutes unless explicitly requested
- **Explainable**: Every suggestion shows why (ROI + rationale)
- **Graceful degradation**: When GPU busy, fall back to lightweight heuristics

---

## Target Architecture

```
Continuous Capture (15-30fps) ─► Ring Buffer ─► Change/Saliency Gate ─► FastVLM + OCR
                                      ↓                                        ↓
                                Frame History               Context Fusion & State Tracker
                                      ↓                                        ↓
                                                         Policy Engine (intervention logic)
                                                                     ↓
                                                      Planner & Tool Router (skills)
                                                                     ↓
                                                     Output Manager (toast/TTS/command)
                                                                     ↓
                                                        Memory System (episodic + RAG)
```

---

## Core Components

### 1. Continuous Screen Capture System
- **Target**: 15-30 fps continuous capture (vs current 0.5 fps event-driven)
- **Ring Buffer**: Last 12-20 seconds in memory
- **Multi-resolution**: Full res for OCR, downscaled for VLM

### 2. Advanced Perception Layer
- **ROI Detection**: Identify regions of interest (errors, modals, active areas)
- **OCR Pipeline**: Extract all text from screen
- **UI Tree**: Parse application structure via Accessibility APIs
- **Saliency Mapping**: Focus on areas with motion/change

### 3. Context Fusion & State Tracking
- **Session Model**: Track current app, file, task hypothesis
- **Event History**: Rolling 30-90 minute episodic buffer
- **Entity Extraction**: Track mentioned files, URLs, people, concepts

### 4. Policy Engine
- **Intervention Logic**: When to speak/notify
- **Risk Assessment**: Detect dangerous actions
- **Utility Scoring**: Measure helpfulness of potential interventions
- **User State**: Detect if typing, in meeting, focused

### 5. Skills & Tool Router
- **Built-in Skills**: Error explanation, summarization, automation
- **Tool Integration**: Clipboard, search, file operations
- **Safety Guards**: Preview destructive operations

### 6. Memory System
- **Episodic Memory**: Time-ordered session summaries
- **Semantic Memory**: Embedded knowledge base
- **Local RAG**: Query project files and documentation

### 7. Output Manager
- **Silent Toast**: Non-intrusive notifications
- **Command Palette**: ⌘⇧K for explicit requests
- **TTS**: Voice output for critical alerts
- **Feedback Loop**: 👍/👎 reactions

### 8. Privacy & Security
- **On-device Only**: No cloud processing
- **Redaction**: Auto-remove secrets before storage
- **Panic Switch**: Instant pause button
- **App Allowlist**: Per-app privacy controls

---

## Implementation Phases

### Phase 1: Foundation Upgrade (Week 1-2)
- Continuous capture system
- Ring buffer implementation
- Basic ROI detection
- FastVLM integration improvements

### Phase 2: Intelligence Layer (Week 3-4)
- OCR pipeline
- Context fusion
- State tracking
- Basic policy engine

### Phase 3: Skills & Memory (Week 5-6)
- Tool router
- Core skills (error, summarize)
- Episodic memory
- Local RAG setup

### Phase 4: UX & Polish (Week 7-8)
- Toast notifications
- Command palette
- TTS integration
- Feedback system

### Phase 5: Privacy & Performance (Week 9-10)
- Redaction system
- Privacy controls
- Performance optimization
- Beta release

---

## Success Metrics

- **Latency**: <300ms end-to-end for interventions
- **Accuracy**: >90% relevant interventions
- **Noise**: <2 false positives per hour
- **Memory**: <3GB peak usage
- **GPU**: <10% duty cycle during normal use

---

## Key Differences from Current Porter.AI

1. **Capture**: Continuous vs event-driven
2. **Analysis**: Multi-layer (VLM + OCR + UI) vs VLM-only
3. **Memory**: Persistent context vs stateless
4. **Intelligence**: Policy-driven vs threshold-based
5. **Interaction**: Multi-modal output vs dashboard-only
6. **Privacy**: Comprehensive controls vs basic
7. **Performance**: Optimized pipeline vs simple loop