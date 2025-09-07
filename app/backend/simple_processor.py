#!/usr/bin/env python3
"""
Lightweight processor for screen analysis without heavy ML models
Uses simple heuristics and pattern matching for memory efficiency
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging
import random
import json
import cv2
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedVLMProcessor:
    """Lightweight processor with pattern matching instead of ML models"""
    
    def __init__(self):
        """Initialize without any ML model loading"""
        self.last_description = ""
        self.description_count = 0
        self.last_brightness = None
        logger.info("SimplifiedVLMProcessor initialized (no ML models)")
        
    async def describe_screen(self, frame: np.ndarray, prompt: Optional[str] = None) -> str:
        """Generate varied descriptions based on visual patterns"""
        if frame is None:
            return "Unable to capture screen"
        
        # Analyze the frame
        avg_brightness = np.mean(frame)
        avg_color = np.mean(frame, axis=(0, 1))
        
        # Detect UI patterns
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate color variance (helps detect content type)
        color_variance = np.std(frame)
        
        # Time-based context
        hour = datetime.now().hour
        time_context = "Morning" if 5 <= hour < 12 else "Afternoon" if 12 <= hour < 17 else "Evening" if 17 <= hour < 22 else "Late night"
        
        # Detect major changes
        brightness_changed = self.last_brightness is not None and abs(avg_brightness - self.last_brightness) > 30
        self.last_brightness = avg_brightness
        
        # Build description based on patterns
        descriptions = []
        
        # Dark theme detection
        if avg_brightness < 100:
            if edge_density > 0.05:  # Lots of edges = code/terminal
                descriptions.extend([
                    f"{time_context} coding session in progress",
                    "Developer working in IDE or terminal",
                    "Programming environment active",
                    "Code editor in focus"
                ])
            else:
                descriptions.extend([
                    "Dark mode application in use",
                    "Working in low-light interface",
                    f"{time_context} work in dark theme"
                ])
        
        # Light theme detection
        elif avg_brightness > 200:
            if color_variance < 30:  # Low variance = document
                descriptions.extend([
                    "Document or text editor open",
                    "Reading or writing content",
                    "Working with text documents",
                    f"{time_context} documentation work"
                ])
            else:
                descriptions.extend([
                    "Browsing web content",
                    "Light-themed application active",
                    "Viewing online resources",
                    f"{time_context} web activity"
                ])
        
        # Mixed brightness
        else:
            if edge_density > 0.08:  # High edge density = complex UI
                descriptions.extend([
                    "Complex application interface",
                    "Multi-panel workspace active",
                    "Working with detailed UI",
                    f"{time_context} productive session"
                ])
            elif color_variance > 60:  # High variance = media/graphics
                descriptions.extend([
                    "Viewing media or graphics",
                    "Visual content on screen",
                    "Multimedia application active",
                    "Working with images or video"
                ])
            else:
                descriptions.extend([
                    f"{time_context} computer usage",
                    "Active desktop session",
                    "General application usage",
                    "Working on computer tasks"
                ])
        
        # Add change detection
        if brightness_changed:
            descriptions.append("Switched to different application")
        
        # Select description (avoid repeating)
        available = [d for d in descriptions if d != self.last_description]
        if not available:
            available = descriptions
        
        description = random.choice(available)
        self.last_description = description
        self.description_count += 1
        
        # Add occasional variety
        if self.description_count % 10 == 0:
            description = f"Continuous {description.lower()}"
        elif self.description_count % 7 == 0:
            description = f"Still {description.lower()}"
        
        return description
            
    async def analyze_for_risks(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect potential risks using enhanced pattern analysis"""
        if frame is None:
            return {'risk_score': 0.0, 'reason': 'No frame', 'suggestion': None}
            
        # Extract color channels
        red_channel = frame[:, :, 0]
        green_channel = frame[:, :, 1]
        blue_channel = frame[:, :, 2]
        
        # Look for error indicators (pure red regions)
        pure_red = (red_channel > 200) & (green_channel < 100) & (blue_channel < 100)
        red_ratio = np.sum(pure_red) / red_channel.size
        
        # Look for warning indicators (yellow/orange)
        yellow_pixels = (red_channel > 200) & (green_channel > 150) & (blue_channel < 100)
        yellow_ratio = np.sum(yellow_pixels) / red_channel.size
        
        # Check screen brightness
        avg_brightness = np.mean(frame)
        
        # Check for sudden changes (potential popups/alerts)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Look for rectangular patterns (dialog boxes)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_rectangles = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:  # Large rectangular area
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approx) == 4:  # Rectangle detected
                    large_rectangles += 1
        
        # Determine risk level
        if red_ratio > 0.02:  # Significant red
            return {
                'risk_score': 0.7,
                'reason': 'Error indicators detected (red alerts)',
                'suggestion': 'Review error messages immediately'
            }
        elif yellow_ratio > 0.02:  # Significant yellow
            return {
                'risk_score': 0.5,
                'reason': 'Warning indicators present',
                'suggestion': 'Check for warnings or alerts'
            }
        elif large_rectangles > 2:  # Multiple dialog boxes
            return {
                'risk_score': 0.4,
                'reason': 'Multiple dialog boxes detected',
                'suggestion': 'Respond to system prompts'
            }
        elif avg_brightness < 10:  # Screen off/black
            return {
                'risk_score': 0.2,
                'reason': 'Screen appears inactive',
                'suggestion': 'Check if system is idle'
            }
        
        # Normal variations
        hour = datetime.now().hour
        if 2 <= hour < 6:
            return {
                'risk_score': 0.15,
                'reason': 'Late night activity',
                'suggestion': 'Consider taking a break'
            }
        
        return {
            'risk_score': 0.1,
            'reason': 'Normal activity patterns',
            'suggestion': None
        }
        
    async def suggest_next_action(self, frame: np.ndarray, context: Optional[str] = None) -> str:
        """Provide helpful suggestions"""
        suggestions = [
            "Save your work",
            "Take a break if you've been working for a while",
            "Check for any pending updates",
            "Review recent changes",
            "Close unused applications to free memory"
        ]
        return random.choice(suggestions)
        
    async def calculate_importance(self, description: str, risk_score: float) -> float:
        """Calculate importance score"""
        # High risk = high importance
        if risk_score > 0.5:
            return 0.7 + (risk_score * 0.3)
            
        # Check for keywords
        important_keywords = ['error', 'failed', 'warning', 'critical', 'password']
        if any(keyword in description.lower() for keyword in important_keywords):
            return 0.6
            
        # Default low importance for normal activity
        return 0.2
        
    def cleanup(self):
        """Clean up (no resources to release)"""
        logger.info("SimplifiedVLMProcessor cleaned up")