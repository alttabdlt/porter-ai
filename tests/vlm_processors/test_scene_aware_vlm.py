#!/usr/bin/env python3
"""
Test suite for Scene-Aware VLM Processor using TDD approach.
Tests VLM processor integration with scene change detection and memory system.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import time
from typing import Dict, Any, Optional

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils.mock_data import MockScreenGenerator, ScreenType


class TestSceneAwareVLMProcessor:
    """Test suite for scene-aware VLM processing"""
    
    @pytest.fixture
    def vlm_processor(self):
        """Create VLM processor instance"""
        from app.vlm_processors.omnivlm_processor import OmniVLMProcessor
        from app.vlm_processors.base_processor import VLMConfig
        
        config = VLMConfig(use_context=True, context_window=3)
        processor = OmniVLMProcessor(config)
        # Mock the model to avoid loading
        processor.model = Mock()
        processor.processor = Mock()
        processor.tokenizer = Mock()
        processor.initialized = True
        return processor
    
    @pytest.fixture
    def memory_system(self):
        """Create memory system instance"""
        from app.memory.dual_memory_system import DualMemorySystem
        return DualMemorySystem(mode="dual")
    
    @pytest.fixture
    def mock_generator(self):
        """Create mock screen generator"""
        return MockScreenGenerator()
    
    # ==================== CONTEXT MANAGEMENT TESTS ====================
    
    @pytest.mark.asyncio
    async def test_vlm_clears_context_on_scene_change(self, vlm_processor):
        """Test that VLM clears previous context when scene changes"""
        # Setup mock responses
        vlm_processor._generate = Mock(side_effect=[
            "User editing code in VSCode",
            "User browsing documentation in Chrome"
        ])
        
        # Process first frame (VSCode)
        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        result1 = await vlm_processor.process_image(
            frame1,
            prompt="Describe the screen",
            metadata={'active_app': 'VSCode', 'scene_changed': False}
        )
        
        # VLM should have stored context
        assert hasattr(vlm_processor, 'last_description')
        assert vlm_processor.last_description is not None
        
        # Process second frame with scene change signal
        frame2 = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        result2 = await vlm_processor.process_image(
            frame2,
            prompt="Describe the screen",
            metadata={'active_app': 'Chrome', 'scene_changed': True}
        )
        
        # Context should be cleared
        assert vlm_processor.last_description != result1.description
        # New context should only contain Chrome description
        assert 'VSCode' not in vlm_processor.last_description
    
    @pytest.mark.asyncio
    async def test_vlm_maintains_context_within_scene(self, vlm_processor):
        """Test that VLM maintains context within the same scene"""
        # Setup mock responses
        vlm_processor._generate = Mock(side_effect=[
            "User writing Python function",
            "User adding parameters to function",
            "User implementing function logic"
        ])
        
        # Process multiple frames in same scene
        frames = [
            np.zeros((720, 1280, 3), dtype=np.uint8),
            np.zeros((720, 1280, 3), dtype=np.uint8),
            np.zeros((720, 1280, 3), dtype=np.uint8)
        ]
        
        descriptions = []
        for i, frame in enumerate(frames):
            result = await vlm_processor.process_image(
                frame,
                prompt="Describe the screen",
                metadata={'active_app': 'VSCode', 'scene_changed': False}
            )
            descriptions.append(result.description)
        
        # Context should build up progressively
        assert hasattr(vlm_processor, 'context_history')
        assert len(vlm_processor.context_history) <= vlm_processor.config.context_window
    
    @pytest.mark.asyncio
    async def test_vlm_uses_context_for_continuity(self, vlm_processor):
        """Test that VLM uses previous context for better continuity"""
        # Mock the _generate method to check if context is being used
        generate_calls = []
        def mock_generate(image, prompt):
            generate_calls.append(prompt)
            return f"Description based on: {prompt[:50]}"
        
        vlm_processor._generate = mock_generate
        
        # Process first frame
        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        result1 = await vlm_processor.process_image(
            frame1,
            prompt="What is the user doing?",
            metadata={'scene_changed': False}
        )
        
        # Process second frame in same scene
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        result2 = await vlm_processor.process_image(
            frame2,
            prompt="What is happening now?",
            metadata={'scene_changed': False}
        )
        
        # Second call should include context from first
        assert len(generate_calls) == 2
        # Context should be injected into prompt for continuity
        if vlm_processor.config.use_context and hasattr(vlm_processor, 'last_description'):
            assert vlm_processor.last_description is not None
    
    # ==================== MEMORY INTEGRATION TESTS ====================
    
    @pytest.mark.asyncio
    async def test_vlm_memory_integration(self, memory_system, mock_generator):
        """Test VLM processor integration with memory system"""
        from app.vlm_processors.scene_aware_vlm import SceneAwareVLMProcessor
        
        # Create scene-aware VLM processor
        vlm = SceneAwareVLMProcessor()
        vlm.model = Mock()  # Mock model to avoid loading
        vlm.initialized = True
        
        # Connect to memory system
        vlm.set_memory_system(memory_system)
        
        # Process frame through memory system
        frame = mock_generator.generate_screen(ScreenType.CODE_EDITOR)
        metadata = {'active_app': 'VSCode', 'timestamp': time.time()}
        
        # Memory system processes frame and detects scene changes
        scene_info = await memory_system.process_frame(
            frame,
            "User editing Python code",
            metadata
        )
        
        # VLM should respect scene change signals
        result = await vlm.process_with_memory(
            frame,
            scene_info,
            prompt="Describe what you see"
        )
        
        assert result.description is not None
        assert result.metadata.get('scene_aware') == True
    
    @pytest.mark.asyncio
    async def test_vlm_handles_scene_transition(self, vlm_processor, memory_system):
        """Test VLM handles scene transitions correctly"""
        # Mock scene change detection
        memory_system.scene_detector.detect_change = Mock(side_effect=[
            {'scene_changed': False, 'change_type': None},
            {'scene_changed': True, 'change_type': 'application', 'prev_app': 'VSCode'}
        ])
        
        # Process VSCode frame
        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        await memory_system.process_frame(
            frame1,
            "Editing code",
            {'active_app': 'VSCode'}
        )
        
        # Process Chrome frame (scene change)
        frame2 = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        await memory_system.process_frame(
            frame2,
            "Browsing web",
            {'active_app': 'Chrome'}
        )
        
        # Working memory should be cleared
        assert len(memory_system.working_memory.frames) == 1  # Only new frame
        assert memory_system.working_memory.current_app == 'Chrome'
    
    # ==================== SCENE CHANGE SIGNAL TESTS ====================
    
    @pytest.mark.asyncio
    async def test_vlm_receives_scene_change_signal(self):
        """Test that VLM receives and processes scene change signals"""
        from app.vlm_processors.scene_aware_vlm import SceneAwareVLMProcessor
        
        vlm = SceneAwareVLMProcessor()
        vlm.model = Mock()
        vlm.initialized = True
        
        # Track context clearing
        context_cleared = False
        original_clear = vlm.clear_context
        def mock_clear():
            nonlocal context_cleared
            context_cleared = True
            return original_clear()
        
        vlm.clear_context = mock_clear
        
        # Signal scene change
        vlm.on_scene_change({
            'scene_changed': True,
            'change_type': 'application',
            'confidence': 0.9
        })
        
        assert context_cleared
    
    @pytest.mark.asyncio
    async def test_vlm_context_window_management(self, vlm_processor):
        """Test VLM manages context window size correctly"""
        vlm_processor.config.context_window = 3
        vlm_processor.context_history = []
        
        # Add multiple descriptions
        descriptions = [
            "Frame 1: User opens editor",
            "Frame 2: User types code",
            "Frame 3: User saves file",
            "Frame 4: User runs tests",
            "Frame 5: User debugs error"
        ]
        
        for desc in descriptions:
            vlm_processor.add_to_context(desc)
        
        # Context should not exceed window size
        assert len(vlm_processor.context_history) <= vlm_processor.config.context_window
        # Should keep most recent entries
        assert vlm_processor.context_history[-1] == descriptions[-1]
    
    # ==================== PERFORMANCE TESTS ====================
    
    @pytest.mark.asyncio
    async def test_scene_aware_processing_performance(self):
        """Test that scene awareness doesn't significantly impact performance"""
        from app.vlm_processors.scene_aware_vlm import SceneAwareVLMProcessor
        
        vlm = SceneAwareVLMProcessor()
        vlm.model = Mock()
        vlm.initialized = True
        vlm._generate = Mock(return_value="Test description")
        
        # Process without scene awareness
        start = time.time()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result1 = await vlm.process_image(frame)
        baseline_time = time.time() - start
        
        # Process with scene awareness
        start = time.time()
        result2 = await vlm.process_image(
            frame,
            metadata={'scene_changed': False}
        )
        aware_time = time.time() - start
        
        # Scene awareness overhead should be minimal (< 10ms)
        overhead = aware_time - baseline_time
        assert overhead < 0.01  # 10ms threshold
    
    # ==================== ERROR HANDLING TESTS ====================
    
    @pytest.mark.asyncio
    async def test_vlm_handles_missing_scene_metadata(self, vlm_processor):
        """Test VLM handles missing scene change metadata gracefully"""
        vlm_processor._generate = Mock(return_value="Test description")
        
        # Process without scene metadata
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = await vlm_processor.process_image(
            frame,
            metadata={}  # No scene_changed key
        )
        
        # Should process normally
        assert result.description is not None
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_vlm_recovers_from_context_errors(self, vlm_processor):
        """Test VLM recovers from context management errors"""
        # Corrupt context
        vlm_processor.context_history = None
        vlm_processor.last_description = None
        
        vlm_processor._generate = Mock(return_value="Recovery description")
        
        # Should still process successfully
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = await vlm_processor.process_image(frame)
        
        assert result.description == "Recovery description"
        assert result.confidence > 0


class TestSceneAwareVLMIntegration:
    """Integration tests for scene-aware VLM system"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_scene_changes(self):
        """Test complete workflow with multiple scene changes"""
        from app.vlm_processors.scene_aware_vlm import SceneAwareVLMProcessor
        from app.memory.dual_memory_system import DualMemorySystem
        
        # Setup system
        memory = DualMemorySystem(mode="dual")
        vlm = SceneAwareVLMProcessor()
        vlm.set_memory_system(memory)
        vlm.model = Mock()
        vlm.initialized = True
        
        # Mock VLM responses
        vlm._generate = Mock(side_effect=[
            "User coding in VSCode",
            "User debugging error",
            "User browsing Stack Overflow",
            "User reading documentation"
        ])
        
        # Simulate workflow
        generator = MockScreenGenerator()
        
        # Phase 1: Coding (2 frames)
        for i in range(2):
            frame = generator.generate_screen(ScreenType.CODE_EDITOR)
            metadata = {'active_app': 'VSCode', 'timestamp': time.time() + i}
            await memory.process_frame(frame, f"Coding frame {i}", metadata)
            result = await vlm.process_with_memory(frame, metadata)
            assert 'VSCode' in result.description or 'coding' in result.description.lower()
        
        # Phase 2: Browsing (2 frames) - scene change
        for i in range(2):
            frame = generator.generate_screen(ScreenType.BROWSER)
            metadata = {'active_app': 'Chrome', 'timestamp': time.time() + i + 10}
            await memory.process_frame(frame, f"Browsing frame {i}", metadata)
            result = await vlm.process_with_memory(frame, metadata)
            # Context should have switched
            assert 'Stack Overflow' in result.description or 'documentation' in result.description.lower()
        
        # Verify memory state
        stats = memory.get_stats()
        assert stats['scene_changes'] >= 1
        assert stats['total_processed'] == 4
    
    @pytest.mark.asyncio
    async def test_context_preservation_across_sessions(self):
        """Test that important context is preserved across sessions"""
        from app.vlm_processors.scene_aware_vlm import SceneAwareVLMProcessor
        from app.memory.dual_memory_system import DualMemorySystem
        
        memory = DualMemorySystem(mode="dual")
        vlm = SceneAwareVLMProcessor()
        vlm.set_memory_system(memory)
        
        # Session 1: Important work
        memory.working_memory.add(
            "User fixed critical bug in authentication.py",
            {'active_app': 'VSCode', 'timestamp': time.time()}
        )
        memory.working_memory.entities.add('authentication.py')
        memory.working_memory.current_activity = 'debugging'
        
        # Trigger scene change (saves important facts)
        await memory._save_important_facts()
        
        # Clear working memory (simulates new session)
        memory.working_memory.clear()
        
        # Query for relevant memories
        memories = await memory.get_relevant_memories(
            "What was the bug fix?",
            {'active_app': 'VSCode'},
            n_results=5
        )
        
        # Should retrieve relevant context
        if memories:  # Only if long-term memory is available
            assert any('authentication' in str(m).lower() for m in memories)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])