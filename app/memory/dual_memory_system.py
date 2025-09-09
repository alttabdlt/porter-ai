#!/usr/bin/env python3
"""
Dual Memory System: Hierarchical memory with Working (ephemeral) and Long-term (persistent) storage.
Prevents context bleeding while maintaining useful memory across sessions.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Set
from collections import deque
import numpy as np
import re
import asyncio

from .scene_change_detector import SceneChangeDetector
from .memory_layer import MemoryLayer

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    Working Memory: Ephemeral storage for current scene/session.
    Resets on scene changes to prevent context bleeding.
    """
    
    def __init__(self, max_frames: int = 5):
        """
        Initialize working memory.
        
        Args:
            max_frames: Maximum number of frames to keep
        """
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)
        self.current_activity = None
        self.current_app = None
        self.entities: Set[str] = set()
        
        # Activity patterns for detection
        self.activity_patterns = {
            'coding': ['code', 'function', 'variable', 'debug', 'edit', 'syntax', 'compile'],
            'browsing': ['search', 'google', 'documentation', 'stackoverflow', 'github'],
            'debugging': ['error', 'bug', 'fix', 'trace', 'breakpoint', 'exception'],
            'testing': ['test', 'pytest', 'unittest', 'assert', 'pass', 'fail'],
            'documenting': ['readme', 'documentation', 'comment', 'docstring', 'markdown'],
            'communicating': ['email', 'chat', 'message', 'slack', 'teams', 'meeting']
        }
        
        logger.info(f"WorkingMemory initialized with max_frames={max_frames}")
    
    def add(self, description: str, metadata: Dict[str, Any]):
        """
        Add a frame to working memory.
        
        Args:
            description: VLM description of the frame
            metadata: Additional metadata (app, timestamp, etc.)
        """
        frame_data = {
            'description': description,
            'metadata': metadata,
            'timestamp': metadata.get('timestamp', time.time())
        }
        
        self.frames.append(frame_data)
        
        # Update current app
        if 'app' in metadata or 'active_app' in metadata:
            self.current_app = metadata.get('app') or metadata.get('active_app')
        
        # Extract entities from description
        self._extract_entities(description)
        
        # Detect activity
        self.current_activity = self._detect_activity()
        
        logger.debug(f"Added frame to working memory: {description[:50]}...")
    
    def clear(self):
        """
        Clear working memory (on scene change).
        Preserves entities for fact extraction.
        """
        self.frames.clear()
        self.current_activity = None
        # Keep entities for extraction before full reset
        logger.info("Working memory cleared (preserving entities)")
    
    def _extract_entities(self, text: str):
        """
        Extract entities (files, functions, variables, etc.) from text.
        
        Args:
            text: Text to extract entities from
        """
        # Extract file names
        file_pattern = r'\b[\w\-]+\.\w+\b'
        files = re.findall(file_pattern, text)
        self.entities.update(files)
        
        # Extract function/class names (CamelCase or snake_case)
        code_pattern = r'\b[A-Z][a-zA-Z0-9_]*\b|\b[a-z]+_[a-z_]+\b'
        code_entities = re.findall(code_pattern, text)
        self.entities.update(code_entities)
        
        # Extract quoted strings (potential important values)
        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted = re.findall(quoted_pattern, text)
        self.entities.update(quoted)
    
    def _detect_activity(self) -> Optional[str]:
        """
        Detect current activity from frame descriptions.
        
        Returns:
            Detected activity type or None
        """
        if not self.frames:
            return None
        
        # Combine recent descriptions
        recent_text = ' '.join(f['description'].lower() for f in list(self.frames)[-3:])
        
        # Score each activity
        activity_scores = {}
        for activity, keywords in self.activity_patterns.items():
            score = sum(1 for keyword in keywords if keyword in recent_text)
            if score > 0:
                activity_scores[activity] = score
        
        # Return highest scoring activity
        if activity_scores:
            return max(activity_scores, key=activity_scores.get)
        
        return None
    
    def extract_key_facts(self) -> Optional[Dict[str, Any]]:
        """
        Extract key facts before clearing memory.
        
        Returns:
            Dictionary of important facts or None if nothing significant
        """
        if not self.entities and not self.current_activity:
            return None
        
        facts = {
            'entities': list(self.entities),
            'activity': self.current_activity,
            'app': self.current_app,
            'summary': self.summarize(),
            'timestamp': time.time()
        }
        
        return facts
    
    def summarize(self) -> str:
        """
        Summarize current session.
        
        Returns:
            Summary string
        """
        if not self.frames:
            return ""
        
        # Get key points from descriptions
        descriptions = [f['description'] for f in self.frames]
        
        # Simple summarization: combine unique key phrases
        key_phrases = []
        for desc in descriptions:
            # Extract action phrases
            if 'edit' in desc.lower():
                key_phrases.append('edited files')
            if 'fix' in desc.lower() or 'bug' in desc.lower():
                key_phrases.append('fixed bugs')
            if 'test' in desc.lower():
                key_phrases.append('ran tests')
            if 'search' in desc.lower():
                key_phrases.append('searched documentation')
            if 'write' in desc.lower() or 'implement' in desc.lower():
                key_phrases.append('wrote code')
        
        if key_phrases:
            return f"User {', '.join(set(key_phrases))} in {self.current_app or 'application'}"
        
        return f"User worked in {self.current_app or 'application'}"
    
    def get_recent(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Get recent frames from working memory.
        
        Args:
            n: Number of recent frames to return
            
        Returns:
            List of recent frames
        """
        return list(self.frames)[-n:]


class DualMemorySystem:
    """
    Dual Memory System: Manages both working and long-term memory.
    Handles scene changes intelligently to prevent context bleeding.
    """
    
    def __init__(
        self,
        mode: str = "dual",
        working_memory_size: int = 5,
        scene_detection_config: Optional[Dict] = None
    ):
        """
        Initialize dual memory system.
        
        Args:
            mode: Memory mode ('dual', 'working_only', 'persistent_only')
            working_memory_size: Size of working memory
            scene_detection_config: Configuration for scene detector
        """
        self.mode = mode
        
        # Initialize components
        self.working_memory = WorkingMemory(max_frames=working_memory_size)
        self.long_term_memory = MemoryLayer() if mode != "working_only" else None
        
        # Scene detection
        scene_config = scene_detection_config or {}
        self.scene_detector = SceneChangeDetector(**scene_config)
        
        # Statistics
        self.total_processed = 0
        self.scene_changes = 0
        self.facts_stored = 0
        
        logger.info(f"DualMemorySystem initialized in {mode} mode")
    
    async def initialize(self):
        """Initialize the memory system components."""
        # Long-term memory initialization happens in MemoryLayer constructor
        logger.info("DualMemorySystem initialized")
    
    async def process_frame(
        self,
        frame: Optional[np.ndarray],
        description: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Process a frame through the dual memory system.
        
        Args:
            frame: Current frame (numpy array)
            description: VLM description
            metadata: Frame metadata (app, timestamp, etc.)
            
        Returns:
            Processed description (may include context from memory)
        """
        self.total_processed += 1
        
        # Handle None frame
        if frame is None:
            logger.warning("Received None frame")
            return description
        
        # Detect scene change
        change_info = self.scene_detector.detect_change(frame, metadata)
        
        if change_info['scene_changed']:
            self.scene_changes += 1
            logger.info(f"Scene change detected: {change_info['change_type']} "
                       f"(confidence={change_info['confidence']:.2f})")
            
            # Extract facts before clearing
            if self.mode != "working_only":
                await self._save_important_facts()
            
            # Clear working memory
            self.working_memory.clear()
        
        # Add current frame to working memory
        self.working_memory.add(description, metadata)
        
        # In persistent_only mode, always store
        if self.mode == "persistent_only" and self.long_term_memory:
            await self.long_term_memory.store_context({
                'description': description,
                'app': metadata.get('active_app'),
                'timestamp': time.time()
            })
        
        # Return clean description (no context bleeding)
        return description
    
    async def _save_important_facts(self):
        """Save important facts from working memory to long-term storage."""
        if not self.long_term_memory:
            return
        
        facts = self.working_memory.extract_key_facts()
        
        if facts and facts.get('summary'):
            try:
                await self.long_term_memory.store_context({
                    'description': facts['summary'],
                    'entities': ','.join(facts['entities']) if facts['entities'] else '',
                    'activity': facts['activity'],
                    'app': facts['app'],
                    'timestamp': facts['timestamp']
                })
                self.facts_stored += 1
                logger.info(f"Stored facts to long-term memory: {facts['summary'][:50]}...")
            except Exception as e:
                logger.error(f"Failed to store facts: {e}")
    
    async def get_relevant_memories(
        self,
        query: str,
        metadata: Dict[str, Any],
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from long-term storage.
        
        Args:
            query: Search query
            metadata: Current context metadata
            n_results: Number of results to return
            
        Returns:
            List of relevant memories
        """
        if not self.long_term_memory or self.mode == "working_only":
            return []
        
        try:
            # Search with app filter if available
            app = metadata.get('active_app')
            memories = await self.long_term_memory.search_similar(query, n_results)
            
            # Filter by app if specified
            if app:
                memories = [m for m in memories if m.get('app') == app]
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'mode': self.mode,
            'total_processed': self.total_processed,
            'scene_changes': self.scene_changes,
            'facts_stored': self.facts_stored,
            'working_memory_frames': len(self.working_memory.frames),
            'current_activity': self.working_memory.current_activity,
            'current_app': self.working_memory.current_app,
            'entities_tracked': len(self.working_memory.entities)
        }
        
        # Add scene detector stats
        scene_stats = self.scene_detector.get_stats()
        stats.update({
            'scene_' + k: v for k, v in scene_stats.items()
        })
        
        return stats
    
    def reset(self):
        """Reset the entire memory system."""
        self.working_memory.clear()
        self.working_memory.entities.clear()
        self.scene_detector.reset()
        self.total_processed = 0
        self.scene_changes = 0
        self.facts_stored = 0
        logger.info("DualMemorySystem reset")


# Utility functions
def create_memory_system(config: Dict[str, Any]) -> DualMemorySystem:
    """
    Create a memory system from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DualMemorySystem
    """
    return DualMemorySystem(
        mode=config.get('mode', 'dual'),
        working_memory_size=config.get('working_memory_size', 5),
        scene_detection_config=config.get('scene_detection', {})
    )


async def test_memory_system():
    """Test the dual memory system."""
    logger.info("Testing Dual Memory System...")
    
    # Create system
    memory = DualMemorySystem()
    await memory.initialize()
    
    # Simulate frames
    frames = [
        (np.zeros((720, 1280, 3), dtype=np.uint8), "Editing code in VSCode", {'active_app': 'VSCode'}),
        (np.zeros((720, 1280, 3), dtype=np.uint8), "Fixed bug in main.py", {'active_app': 'VSCode'}),
        (np.ones((720, 1280, 3), dtype=np.uint8) * 255, "Browsing documentation", {'active_app': 'Chrome'}),
    ]
    
    for frame, desc, meta in frames:
        result = await memory.process_frame(frame, desc, meta)
        logger.info(f"Processed: {result}")
    
    # Get stats
    stats = memory.get_stats()
    logger.info(f"Stats: {stats}")
    
    logger.info("✅ Dual Memory System test complete!")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_memory_system())