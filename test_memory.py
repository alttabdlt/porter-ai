#!/usr/bin/env python3
"""Test memory usage of imports"""

import psutil
import time

def get_memory():
    return psutil.Process().memory_info().rss / 1024 / 1024

print(f"Start: {get_memory():.1f}MB")

# Core Python imports
import asyncio
import logging
import sys
from pathlib import Path
import webbrowser
import argparse
import base64
from io import BytesIO
import time
import hashlib
import gc
import weakref
print(f"After Python stdlib: {get_memory():.1f}MB")

# Numpy
import numpy as np
print(f"After numpy: {get_memory():.1f}MB")

# PIL
from PIL import Image
print(f"After PIL: {get_memory():.1f}MB")

# OpenCV
import cv2
print(f"After cv2: {get_memory():.1f}MB")

# MSS
import mss
print(f"After mss: {get_memory():.1f}MB")

# psutil
import psutil
print(f"After psutil: {get_memory():.1f}MB")

# Test actual imports from our modules
print("\n--- Testing our module imports ---")

# Import continuous capture
from app.backend.continuous_capture import ContinuousScreenCapture
print(f"After ContinuousScreenCapture: {get_memory():.1f}MB")

# Import simple processor
from app.backend.simple_processor import SimplifiedVLMProcessor
print(f"After SimplifiedVLMProcessor: {get_memory():.1f}MB")

# Import server
from app.backend.server import DashboardServer
print(f"After DashboardServer: {get_memory():.1f}MB")

# Import memory config
from app.backend.memory_safety_config import MemorySafetyConfig, MemoryMonitor
print(f"After MemorySafetyConfig: {get_memory():.1f}MB")

print(f"\n--- Final memory: {get_memory():.1f}MB ---")