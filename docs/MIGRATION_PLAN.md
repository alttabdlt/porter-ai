# Porter.AI → FASTVLM Jarvis Migration Plan

## Current State Analysis

### Existing Files & Their Roles
- `real_screen_capture.py` - Event-driven capture (2+ sec), SSIM detection
- `real_assistant.py` - Main loop, basic importance scoring
- `vlm_processor.py` - FastVLM integration for descriptions
- `run_real.py` - Entry point with dashboard
- `simple_server.py` - WebSocket server
- `simple_dashboard.html` - Basic monitoring UI

### Missing Critical Components
- ❌ Continuous capture (currently event-driven)
- ❌ Ring buffer for frame history
- ❌ OCR extraction
- ❌ UI tree/Accessibility API integration  
- ❌ Context fusion & state tracking
- ❌ Policy engine for interventions
- ❌ Memory system (episodic + semantic)
- ❌ Tool router & skills
- ❌ Toast notifications
- ❌ TTS output
- ❌ Command palette
- ❌ Privacy controls

---

## File-by-File Modifications

### 📄 `real_screen_capture.py` - MAJOR REWRITE

**Current**: Event-driven, captures on change only
**Target**: Continuous capture with ring buffer

```python
# REMOVE:
- get_if_changed() method
- SSIM-only change detection
- 2-second minimum interval

# MODIFY:
class ContinuousScreenCapture:  # Rename from RealScreenCapture
    def __init__(self):
        self.fps = 15  # Up from 0.5 fps
        self.ring_buffer = deque(maxlen=300)  # 20 seconds @ 15fps
        self.capture_thread = None
        self.is_running = False
        
# ADD:
- continuous_capture_loop() - Background thread
- get_frame_window(start_time, end_time) - Query ring buffer
- extract_roi(frame, bbox) - Region extraction
- get_motion_map() - Optical flow for saliency
```

### 📄 `vlm_processor.py` - EXTEND

**Current**: Basic describe_screen()
**Target**: Multi-prompt templates, batched ROI processing

```python
# ADD:
class VLMPromptTemplates:
    DESCRIBE = "Summarize user's current objective"
    HAZARD = "Identify risky actions. Rate 0-1"
    NEXT_STEP = "Suggest next step. 1 sentence"
    EXTRACT = "List URLs, errors, entities. JSON"

# MODIFY:
async def process_rois(self, rois: List[np.ndarray], template: str)
    # Batch multiple regions in one call
    
# ADD:
- extract_entities(frame) -> Dict
- detect_errors(frame) -> List[str]
- suggest_action(frame, context) -> str
```

### 📄 `real_assistant.py` - MAJOR EXTENSION

**Current**: Simple process loop
**Target**: Complex state machine with context

```python
# ADD:
class SessionContext:
    current_app: str
    current_file: str
    task_hypothesis: str
    entities: Set[str]
    last_intervention: float
    
class PolicyEngine:
    def should_intervene(self, context, event) -> bool
    def calculate_utility(self, suggestion) -> float
    def check_risk_level(self, screen) -> float

# MODIFY:
- Add context tracking to main loop
- Replace simple threshold with policy engine
- Add intervention queue management
```

### 📄 NEW: `ocr_engine.py`

```python
class OCREngine:
    def __init__(self):
        self.tesseract = None  # Or Vision.framework on macOS
        
    def extract_all_text(self, frame) -> List[TextRegion]
    def find_text_regions(self, frame) -> List[BBox]
    def extract_structured_data(self, text) -> Dict
```

### 📄 NEW: `memory_system.py`

```python
class MemorySystem:
    def __init__(self):
        self.episodic = deque(maxlen=1000)  # Last N events
        self.semantic = ChromaDB()  # Or similar
        
    def store_event(self, event: ScreenEvent)
    def query_similar(self, query: str, k=5)
    def summarize_session(self) -> str
    def get_context_window(self, minutes=30) -> List[Event]
```

### 📄 NEW: `context_fusion.py`

```python
class ContextFusion:
    def __init__(self):
        self.state_tracker = StateTracker()
        self.app_adapters = {}  # Per-app context extractors
        
    def update_context(self, frame, ocr_text, vlm_desc) -> SessionContext
    def detect_task_switch(self) -> bool
    def extract_entities(self, text) -> Set[str]
```

### 📄 NEW: `policy_engine.py`

```python
class PolicyEngine:
    def __init__(self):
        self.thresholds = {
            'risk': 0.7,
            'utility': 0.5,
            'novelty': 0.3,
            'time_between': 60  # seconds
        }
        
    def evaluate_intervention(self, context, suggestion) -> Decision
    def is_user_interruptible(self) -> bool
    def calculate_scores(self, event) -> Dict[str, float]
```

### 📄 NEW: `tool_router.py`

```python
class ToolRouter:
    def __init__(self):
        self.skills = {
            'explain_error': ExplainErrorSkill(),
            'summarize': SummarizeSkill(),
            'clipboard': ClipboardSkill(),
        }
        
    def route_intent(self, intent: str, args: Dict) -> ToolResult
    def execute_skill(self, skill_name: str, context: Dict)
```

### 📄 NEW: `output_manager.py`

```python
class OutputManager:
    def __init__(self):
        self.toast_queue = Queue()
        self.tts_engine = None  # pyttsx3 or macOS Speech
        
    def show_toast(self, message: str, priority: str)
    def speak(self, text: str, interrupt: bool = False)
    def show_command_palette(self)
```

### 📄 NEW: `privacy_manager.py`

```python
class PrivacyManager:
    def __init__(self):
        self.redaction_patterns = []  # Regex for secrets
        self.app_allowlist = set()
        self.paused = False
        
    def redact_sensitive(self, text: str) -> str
    def should_process_app(self, app_name: str) -> bool
    def pause_monitoring(self, duration: int)
```

### 📄 `run_real.py` - MODIFY

```python
# ADD:
- Command line args for continuous mode
- Memory system initialization
- Policy engine setup
- Privacy manager init

# MODIFY:
- Main loop to coordinate all components
- Add keyboard shortcuts (Cmd+Shift+K)
```

### 📄 `simple_dashboard.html` - EXTEND

```html
<!-- ADD: -->
- Command palette overlay
- Toast notification container
- Privacy controls panel
- Memory usage indicator
- Intervention history
```

---

## Micro-Action Checklist

### Week 1: Foundation
- [ ] Fork `real_screen_capture.py` → `continuous_capture.py`
- [ ] Implement ring buffer with deque
- [ ] Add continuous capture thread (15 fps)
- [ ] Create frame windowing methods
- [ ] Add motion detection for saliency
- [ ] Test memory usage with ring buffer
- [ ] Add FPS throttling based on GPU load

### Week 2: Perception
- [ ] Create `ocr_engine.py` with Tesseract/Vision.framework
- [ ] Implement text region detection
- [ ] Add OCR text extraction pipeline
- [ ] Create `accessibility_bridge.py` for UI tree
- [ ] Implement ROI batching in `vlm_processor.py`
- [ ] Add new VLM prompt templates
- [ ] Test OCR + VLM fusion

### Week 3: Context & State
- [ ] Create `context_fusion.py`
- [ ] Implement SessionContext dataclass
- [ ] Add state tracking to assistant loop
- [ ] Create app-specific adapters (browser, IDE)
- [ ] Implement entity extraction
- [ ] Add task hypothesis tracking
- [ ] Test context persistence

### Week 4: Policy & Interventions  
- [ ] Create `policy_engine.py`
- [ ] Implement intervention scoring
- [ ] Add user state detection (typing, meeting)
- [ ] Create intervention queue
- [ ] Add time-based suppression
- [ ] Implement risk assessment
- [ ] Test false positive rate

### Week 5: Memory System
- [ ] Create `memory_system.py`
- [ ] Implement episodic buffer
- [ ] Add ChromaDB for semantic store
- [ ] Create session summarization
- [ ] Implement similarity search
- [ ] Add memory pruning
- [ ] Test retrieval accuracy

### Week 6: Skills & Tools
- [ ] Create `tool_router.py`
- [ ] Implement ExplainErrorSkill
- [ ] Add SummarizeSkill
- [ ] Create ClipboardSkill
- [ ] Add safety guards for destructive ops
- [ ] Implement dry-run previews
- [ ] Test skill routing

### Week 7: Output & UX
- [ ] Create `output_manager.py`
- [ ] Implement toast notifications
- [ ] Add TTS with pyttsx3/macOS Speech
- [ ] Create command palette (Cmd+Shift+K)
- [ ] Add feedback buttons (👍/👎)
- [ ] Implement snooze functionality
- [ ] Test notification timing

### Week 8: Privacy & Security
- [ ] Create `privacy_manager.py`
- [ ] Implement secret redaction
- [ ] Add app allowlist/blocklist
- [ ] Create panic switch
- [ ] Add audit logging
- [ ] Implement data encryption
- [ ] Test redaction effectiveness

### Week 9: Integration & Performance
- [ ] Integrate all components in `run_real.py`
- [ ] Optimize pipeline for <300ms latency
- [ ] Add GPU throttling
- [ ] Implement model prewarming
- [ ] Add caching layer
- [ ] Profile memory usage
- [ ] Fix bottlenecks

### Week 10: Polish & Release
- [ ] Update dashboard with new features
- [ ] Add onboarding flow
- [ ] Create settings UI
- [ ] Write user documentation
- [ ] Add telemetry (opt-in)
- [ ] Package for distribution
- [ ] Beta testing

---

## Dependencies to Add

```txt
# requirements.txt additions
tesseract  # Or pytesseract for OCR
chromadb  # Vector database for memory
pyttsx3  # Text-to-speech
pyobjc  # macOS Accessibility APIs
keyboard  # Global hotkeys
plyer  # Cross-platform notifications
sentence-transformers  # Embeddings
```

---

## Migration Strategy

### Phase 1: Parallel Development
- Keep existing Porter.AI running
- Develop new components in separate files
- Test components individually

### Phase 2: Integration
- Create `run_jarvis.py` as new entry point
- Gradually migrate features
- A/B test old vs new

### Phase 3: Cutover
- Replace `run_real.py` with Jarvis version
- Archive old components
- Full system testing

---

## Risk Mitigation

### Performance Risks
- **Ring buffer memory**: Cap at 20 seconds
- **GPU contention**: Throttle to 10% duty cycle
- **Latency**: Async pipeline, caching

### Privacy Risks
- **Data leakage**: All processing on-device
- **Screenshot storage**: Ephemeral only
- **Secret exposure**: Aggressive redaction

### UX Risks
- **Over-interruption**: Strict rate limiting
- **False positives**: Conservative thresholds
- **User trust**: Full transparency, easy disable

---

## Success Criteria

- [ ] Continuous capture working at 15+ fps
- [ ] OCR extracting >95% of screen text
- [ ] Context tracking across app switches
- [ ] <2 false interventions per hour
- [ ] <300ms intervention latency
- [ ] <3GB memory usage
- [ ] Privacy controls functional
- [ ] Beta user satisfaction >4/5