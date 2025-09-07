#!/usr/bin/env python3
"""
Porter.AI - FASTVLM Jarvis
Simple startup script for the monitoring system
"""

import sys
import os
from pathlib import Path

# Add app/backend to path for imports
backend_path = Path(__file__).parent / 'app' / 'backend'
sys.path.insert(0, str(backend_path))

# Import and run main
from main import main

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 PORTER.AI - FASTVLM JARVIS")
    print("="*70)
    print("\nStarting intelligent screen monitoring system...")
    print("Dashboard will be available at: http://localhost:8000")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    exit(main())