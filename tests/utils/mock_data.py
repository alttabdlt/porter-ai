#!/usr/bin/env python3
"""
Mock data generators for testing Porter.AI components.
Provides realistic test data without requiring actual screen capture or models.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict, Any
import random
import time
from dataclasses import dataclass
from enum import Enum


class ScreenType(Enum):
    """Types of screens to generate"""
    CODE_EDITOR = "code_editor"
    TERMINAL = "terminal"
    BROWSER = "browser"
    EMAIL = "email"
    DOCUMENT = "document"
    DESKTOP = "desktop"
    ERROR_DIALOG = "error_dialog"
    VIDEO_CALL = "video_call"


@dataclass
class MockScreenConfig:
    """Configuration for mock screen generation"""
    width: int = 1920
    height: int = 1080
    screen_type: ScreenType = ScreenType.DESKTOP
    add_noise: bool = False
    add_ui_elements: bool = True
    seed: Optional[int] = None


class MockScreenGenerator:
    """Generate realistic mock screens for testing"""
    
    def __init__(self, config: Optional[MockScreenConfig] = None):
        """Initialize the mock screen generator"""
        self.config = config or MockScreenConfig()
        if self.config.seed:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
    
    def generate_screen(self, screen_type: Optional[ScreenType] = None) -> np.ndarray:
        """
        Generate a mock screen of specified type.
        
        Args:
            screen_type: Type of screen to generate
            
        Returns:
            numpy array representing the screen (RGB)
        """
        screen_type = screen_type or self.config.screen_type
        
        # Create base image
        img = Image.new('RGB', (self.config.width, self.config.height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Generate based on type
        if screen_type == ScreenType.CODE_EDITOR:
            self._draw_code_editor(img, draw)
        elif screen_type == ScreenType.TERMINAL:
            self._draw_terminal(img, draw)
        elif screen_type == ScreenType.BROWSER:
            self._draw_browser(img, draw)
        elif screen_type == ScreenType.EMAIL:
            self._draw_email(img, draw)
        elif screen_type == ScreenType.DOCUMENT:
            self._draw_document(img, draw)
        elif screen_type == ScreenType.ERROR_DIALOG:
            self._draw_error_dialog(img, draw)
        elif screen_type == ScreenType.VIDEO_CALL:
            self._draw_video_call(img, draw)
        else:
            self._draw_desktop(img, draw)
        
        # Convert to numpy array
        screen = np.array(img)
        
        # Add noise if configured
        if self.config.add_noise:
            noise = np.random.randint(-10, 10, screen.shape, dtype=np.int16)
            screen = np.clip(screen.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return screen
    
    def _draw_code_editor(self, img: Image.Image, draw: ImageDraw.Draw):
        """Draw a code editor interface"""
        # Dark background
        draw.rectangle([(0, 0), (self.config.width, self.config.height)], fill=(30, 30, 30))
        
        # Title bar
        draw.rectangle([(0, 0), (self.config.width, 30)], fill=(45, 45, 45))
        draw.text((10, 5), "main.py - Visual Studio Code", fill=(200, 200, 200))
        
        # Sidebar
        draw.rectangle([(0, 30), (200, self.config.height)], fill=(37, 37, 38))
        
        # Code area with syntax highlighting
        y_pos = 50
        code_lines = [
            ("def ", (86, 156, 214)),
            ("process_image", (220, 220, 170)),
            ("(image: np.ndarray):", (255, 255, 255)),
            ("    # Process the image", (106, 153, 85)),
            ("    result = model.predict(image)", (255, 255, 255)),
            ("    return result", (197, 134, 192))
        ]
        
        for line, color in code_lines:
            draw.text((220, y_pos), line, fill=color)
            y_pos += 25
        
        # Line numbers
        for i in range(1, 20):
            draw.text((205, 50 + (i-1) * 25), str(i), fill=(133, 133, 133))
    
    def _draw_terminal(self, img: Image.Image, draw: ImageDraw.Draw):
        """Draw a terminal interface"""
        # Black background
        draw.rectangle([(0, 0), (self.config.width, self.config.height)], fill=(0, 0, 0))
        
        # Terminal output
        y_pos = 20
        lines = [
            "$ python test_cross_platform.py",
            "================================================================",
            "   🚀 PORTER.AI CROSS-PLATFORM TEST SUITE",
            "================================================================",
            "",
            "🖥️  PLATFORM DETECTION TEST",
            "Operating System: Linux",
            "Platform: Linux-5.15.0-azure-x86_64",
            "✅ Platform detection complete",
            "",
            "📸 SCREEN CAPTURE TEST",
            "Found 2 monitor(s):",
            "  Monitor 0: 1920x1080",
            "  Monitor 1: 1920x1080",
            "✅ Capture initialized successfully"
        ]
        
        for line in lines:
            color = (0, 255, 0) if line.startswith("✅") else (255, 255, 255)
            draw.text((20, y_pos), line, fill=color)
            y_pos += 20
    
    def _draw_browser(self, img: Image.Image, draw: ImageDraw.Draw):
        """Draw a browser interface"""
        # White background
        draw.rectangle([(0, 0), (self.config.width, self.config.height)], fill=(255, 255, 255))
        
        # Browser chrome
        draw.rectangle([(0, 0), (self.config.width, 80)], fill=(240, 240, 240))
        
        # URL bar
        draw.rectangle([(100, 25), (self.config.width - 100, 55)], fill=(255, 255, 255))
        draw.text((110, 32), "https://github.com/porter-ai/porter", fill=(0, 0, 0))
        
        # Page content
        draw.rectangle([(50, 120), (self.config.width - 50, 200)], fill=(36, 41, 46))
        draw.text((60, 140), "Porter.AI - Real-Time Screen Intelligence", fill=(255, 255, 255))
        
        # Content blocks
        for i in range(3):
            y = 250 + i * 150
            draw.rectangle([(50, y), (self.config.width - 50, y + 100)], fill=(250, 250, 250))
    
    def _draw_email(self, img: Image.Image, draw: ImageDraw.Draw):
        """Draw an email client interface"""
        draw.rectangle([(0, 0), (self.config.width, self.config.height)], fill=(245, 245, 245))
        
        # Sidebar
        draw.rectangle([(0, 0), (250, self.config.height)], fill=(52, 73, 94))
        draw.text((20, 20), "Inbox (15)", fill=(255, 255, 255))
        draw.text((20, 50), "Sent", fill=(189, 195, 199))
        draw.text((20, 80), "Drafts (2)", fill=(189, 195, 199))
        
        # Email list
        for i in range(5):
            y = 100 + i * 80
            draw.rectangle([(260, y), (self.config.width - 10, y + 70)], fill=(255, 255, 255))
            draw.text((270, y + 10), f"Email Subject {i+1}", fill=(0, 0, 0))
            draw.text((270, y + 35), f"sender{i+1}@example.com", fill=(128, 128, 128))
    
    def _draw_document(self, img: Image.Image, draw: ImageDraw.Draw):
        """Draw a document editor interface"""
        draw.rectangle([(0, 0), (self.config.width, self.config.height)], fill=(255, 255, 255))
        
        # Toolbar
        draw.rectangle([(0, 0), (self.config.width, 60)], fill=(240, 240, 240))
        
        # Document content
        margin = 100
        y_pos = 100
        
        # Title
        draw.text((margin, y_pos), "Project Report", fill=(0, 0, 0))
        y_pos += 40
        
        # Paragraphs
        for i in range(5):
            text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 3
            draw.text((margin, y_pos), text[:100], fill=(0, 0, 0))
            y_pos += 30
    
    def _draw_error_dialog(self, img: Image.Image, draw: ImageDraw.Draw):
        """Draw an error dialog"""
        # Desktop background
        draw.rectangle([(0, 0), (self.config.width, self.config.height)], fill=(58, 110, 165))
        
        # Error dialog
        dialog_width = 500
        dialog_height = 200
        x = (self.config.width - dialog_width) // 2
        y = (self.config.height - dialog_height) // 2
        
        # Dialog box
        draw.rectangle([(x, y), (x + dialog_width, y + dialog_height)], fill=(255, 255, 255))
        draw.rectangle([(x, y), (x + dialog_width, y + 40)], fill=(255, 0, 0))
        
        # Error text
        draw.text((x + 10, y + 10), "⚠️ Error", fill=(255, 255, 255))
        draw.text((x + 20, y + 60), "An unexpected error has occurred.", fill=(0, 0, 0))
        draw.text((x + 20, y + 90), "Error Code: 0x800F0922", fill=(128, 128, 128))
        
        # Buttons
        draw.rectangle([(x + 300, y + 150), (x + 380, y + 180)], fill=(200, 200, 200))
        draw.text((x + 330, y + 158), "OK", fill=(0, 0, 0))
    
    def _draw_video_call(self, img: Image.Image, draw: ImageDraw.Draw):
        """Draw a video call interface"""
        draw.rectangle([(0, 0), (self.config.width, self.config.height)], fill=(30, 30, 30))
        
        # Main video area
        main_video = [(100, 50), (self.config.width - 100, self.config.height - 150)]
        draw.rectangle(main_video, fill=(50, 50, 50))
        draw.text((self.config.width // 2 - 50, self.config.height // 2), 
                 "Video Feed", fill=(150, 150, 150))
        
        # Participant thumbnails
        for i in range(4):
            x = 100 + i * 170
            y = self.config.height - 130
            draw.rectangle([(x, y), (x + 150, y + 100)], fill=(70, 70, 70))
            draw.text((x + 50, y + 40), f"User {i+1}", fill=(200, 200, 200))
        
        # Controls
        controls_y = self.config.height - 130
        draw.ellipse([(self.config.width - 250, controls_y), 
                     (self.config.width - 200, controls_y + 50)], fill=(255, 0, 0))
        draw.ellipse([(self.config.width - 180, controls_y), 
                     (self.config.width - 130, controls_y + 50)], fill=(128, 128, 128))
    
    def _draw_desktop(self, img: Image.Image, draw: ImageDraw.Draw):
        """Draw a desktop with icons"""
        # Gradient background
        for y in range(self.config.height):
            color_value = int(58 + (165 - 58) * y / self.config.height)
            draw.rectangle([(0, y), (self.config.width, y + 1)], 
                          fill=(58, 110, color_value))
        
        # Taskbar
        draw.rectangle([(0, self.config.height - 40), 
                       (self.config.width, self.config.height)], fill=(48, 48, 48))
        
        # Desktop icons
        icons = ["📁 Documents", "💻 Terminal", "🌐 Browser", "📧 Mail"]
        for i, icon in enumerate(icons):
            x = 20
            y = 20 + i * 100
            draw.rectangle([(x, y), (x + 80, y + 80)], fill=(255, 255, 255, 128))
            draw.text((x + 5, y + 85), icon, fill=(255, 255, 255))
    
    def generate_batch(self, count: int, varied: bool = True) -> List[np.ndarray]:
        """
        Generate a batch of mock screens.
        
        Args:
            count: Number of screens to generate
            varied: Whether to vary screen types
            
        Returns:
            List of screen arrays
        """
        screens = []
        screen_types = list(ScreenType) if varied else [self.config.screen_type]
        
        for i in range(count):
            screen_type = random.choice(screen_types)
            screen = self.generate_screen(screen_type)
            screens.append(screen)
        
        return screens


class MockVLMOutput:
    """Generate mock VLM outputs for testing"""
    
    @staticmethod
    def generate_description(screen_type: ScreenType) -> str:
        """Generate realistic VLM description based on screen type"""
        descriptions = {
            ScreenType.CODE_EDITOR: [
                "User is editing Python code in Visual Studio Code with syntax highlighting",
                "Code editor showing a function definition with multiple parameters",
                "IDE displaying Python script with comments and function definitions"
            ],
            ScreenType.TERMINAL: [
                "Terminal showing test execution output with green success indicators",
                "Command line interface displaying Python test results",
                "Console output showing successful test suite execution"
            ],
            ScreenType.BROWSER: [
                "Web browser displaying GitHub repository page",
                "Browser showing project documentation with navigation sidebar",
                "Chrome browser on GitHub with repository statistics visible"
            ],
            ScreenType.EMAIL: [
                "Email client with inbox showing 15 unread messages",
                "Mail application displaying list of recent emails in inbox",
                "Email interface with sidebar navigation and message preview"
            ],
            ScreenType.DOCUMENT: [
                "Document editor showing project report with formatted text",
                "Word processor displaying multi-page document with headers",
                "Text editor with report content and formatting toolbar"
            ],
            ScreenType.ERROR_DIALOG: [
                "Error dialog box showing unexpected error with code 0x800F0922",
                "System error message displayed in modal dialog",
                "Warning dialog with error details and OK button"
            ],
            ScreenType.VIDEO_CALL: [
                "Video conference with 4 participants and screen sharing active",
                "Video call interface showing main speaker and participant thumbnails",
                "Online meeting with muted microphone and active camera"
            ],
            ScreenType.DESKTOP: [
                "Desktop with taskbar and four application icons visible",
                "Windows desktop showing file explorer and application shortcuts",
                "Clean desktop with gradient background and system tray"
            ]
        }
        
        return random.choice(descriptions.get(screen_type, ["Generic screen content"]))
    
    @staticmethod
    def generate_risk_assessment() -> Dict[str, Any]:
        """Generate mock risk assessment"""
        risk_levels = ["low", "medium", "high"]
        risk_level = random.choice(risk_levels)
        
        reasons = {
            "low": "No sensitive information or risky actions detected",
            "medium": "Unsaved changes detected in active document",
            "high": "Terminal command with potential destructive operation"
        }
        
        return {
            "risk_level": risk_level,
            "reason": reasons[risk_level],
            "confidence": random.uniform(0.7, 0.95)
        }
    
    @staticmethod
    def generate_ui_elements() -> Dict[str, List[str]]:
        """Generate mock UI element detection"""
        return {
            "buttons": ["Save", "Cancel", "OK", "Apply"],
            "text_fields": ["Username", "Password", "Search"],
            "menus": ["File", "Edit", "View", "Help"],
            "links": ["https://example.com", "Documentation", "Settings"],
            "errors": [] if random.random() > 0.3 else ["Error: Invalid input"]
        }


class MockMetrics:
    """Generate mock performance metrics"""
    
    @staticmethod
    def generate_metrics(base_fps: float = 10.0) -> Dict[str, float]:
        """Generate realistic performance metrics"""
        return {
            "fps": base_fps + random.uniform(-2, 2),
            "latency": random.uniform(50, 150),  # ms
            "cpu": random.uniform(10, 40),  # percentage
            "memory": random.uniform(500, 2000),  # MB
            "frames_processed": random.randint(100, 10000),
            "frames_analyzed": random.randint(50, 5000)
        }
    
    @staticmethod
    def generate_time_series(duration: int = 60, interval: int = 1) -> List[Dict]:
        """Generate time series metrics data"""
        metrics = []
        base_fps = 10.0
        
        for i in range(0, duration, interval):
            metric = MockMetrics.generate_metrics(base_fps)
            metric["timestamp"] = time.time() + i
            metrics.append(metric)
            
            # Add some variance over time
            base_fps += random.uniform(-0.5, 0.5)
            base_fps = max(5.0, min(15.0, base_fps))
        
        return metrics


# Utility functions
def create_test_image(width: int = 1920, height: int = 1080, 
                     color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """Create a simple test image"""
    return np.full((height, width, 3), color, dtype=np.uint8)


def create_test_pattern(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a test pattern image"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Checkerboard pattern
    block_size = 50
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            if (x // block_size + y // block_size) % 2 == 0:
                img[y:y+block_size, x:x+block_size] = [255, 255, 255]
    
    return img


def create_gradient_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a gradient test image"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        value = int(255 * y / height)
        img[y, :] = [value, value, value]
    
    return img


if __name__ == "__main__":
    # Test the mock data generators
    print("Testing mock data generators...")
    
    # Test screen generator
    generator = MockScreenGenerator()
    
    for screen_type in ScreenType:
        screen = generator.generate_screen(screen_type)
        print(f"Generated {screen_type.value}: {screen.shape}")
    
    # Test batch generation
    batch = generator.generate_batch(5, varied=True)
    print(f"\nGenerated batch of {len(batch)} screens")
    
    # Test VLM output
    for screen_type in ScreenType:
        description = MockVLMOutput.generate_description(screen_type)
        print(f"\n{screen_type.value}: {description}")
    
    # Test metrics
    metrics = MockMetrics.generate_metrics()
    print(f"\nGenerated metrics: {metrics}")
    
    print("\n✅ Mock data generators working correctly!")