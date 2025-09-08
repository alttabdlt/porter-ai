# Display Selection for Multi-Monitor Support

## Overview
Porter.AI now supports individual display selection for multi-monitor setups. Previously, all connected displays were captured as one ultra-wide view, which reduced VLM accuracy. With the new display selection feature, you can capture and analyze content from a specific monitor.

## Features
- **Automatic Display Detection**: Enumerates all connected displays on startup
- **Individual Display Selection**: Choose which display to capture via command-line argument
- **Fallback Handling**: Automatically falls back to primary display if invalid index is provided
- **Real-time Display Info**: Shows display count and resolution during initialization

## Usage

### Basic Usage (Primary Display)
```bash
python app/streaming/main_streaming.py
```
By default, captures from display 0 (primary monitor).

### Select Specific Display
```bash
# Capture from display 0 (primary)
python app/streaming/main_streaming.py --display 0

# Capture from display 1 (secondary)
python app/streaming/main_streaming.py --display 1

# Capture from display 2 (tertiary)
python app/streaming/main_streaming.py --display 2
```

### Command-Line Arguments
- `--display <index>`: Display index to capture (default: 0)
- `--fps <number>`: Frames per second (default: 10)
- `--width <pixels>`: Capture width (default: 1920)
- `--height <pixels>`: Capture height (default: 1080)
- `--no-vlm`: Disable VLM processing

## Display Enumeration
When the streaming pipeline starts, it:
1. Detects all connected displays
2. Lists each display with its index and resolution
3. Selects the requested display or falls back to display 0

Example output:
```
Found 3 display(s):
  Display 0: 1920x1080
  Display 1: 2560x1440
  Display 2: 1920x1080
Using display 1: 2560x1440
```

## Implementation Details

### Architecture
The display selection is implemented in `SimpleScreenCapture` class:
- `display_index` parameter in constructor
- `enumerate_displays()` method for listing all displays
- Automatic fallback logic for invalid indices

### Key Components
1. **SimpleScreenCapture** (`app/streaming/simple_screencapture.py`)
   - Handles display enumeration and selection
   - Manages ScreenCaptureKit stream initialization

2. **StreamingPipeline** (`app/streaming/main_streaming.py`)
   - Accepts display_index configuration
   - Passes display selection to capture component

3. **Command-Line Interface** (`app/streaming/main_streaming.py`)
   - Parses --display argument
   - Validates display index input

## Testing
Comprehensive test coverage in `tests/streaming/test_display_selection.py`:
- Display enumeration tests
- Display selection by index
- Invalid index fallback handling
- Command-line argument parsing
- Integration tests with full pipeline

Run tests:
```bash
python tests/streaming/test_display_selection.py
```

## Benefits
1. **Improved Accuracy**: VLM analyzes content from a single display at native resolution
2. **Flexible Monitoring**: Choose which display to monitor based on your workflow
3. **Better Performance**: Reduced processing overhead by capturing only needed display
4. **Multi-Workflow Support**: Different displays can be monitored for different purposes

## Troubleshooting

### Display Not Found
If you see "Display index X not found", the system will automatically fall back to display 0. Check your display count with:
```bash
python app/streaming/main_streaming.py 2>&1 | grep "Found.*display"
```

### macOS Permissions
Ensure Porter.AI has screen recording permissions:
1. System Preferences → Security & Privacy → Privacy
2. Select "Screen Recording" from the list
3. Check the box next to Terminal or your Python executable

## Future Enhancements
- [ ] Display hot-swapping support
- [ ] Multiple display capture in parallel
- [ ] Display-specific configurations
- [ ] Automatic display selection based on active window
- [ ] Display arrangement visualization

## API Reference

### SimpleScreenCapture
```python
capture = SimpleScreenCapture(fps=10, display_index=1)
displays = capture.enumerate_displays(content)  # List all displays
await capture.initialize()  # Initialize with selected display
```

### StreamingPipeline
```python
pipeline = StreamingPipeline({
    'display_index': 1,  # Select display 1
    'fps': 10,
    'width': 1920,
    'height': 1080
})
```