#!/usr/bin/env python3
"""
Integration tests for the complete scene-aware system.
Tests the interaction between scene detection, memory management, and VLM processing.
"""

import pytest
import numpy as np
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils.mock_data import MockScreenGenerator, ScreenType


class TestSceneAwareSystemIntegration:
    """Integration tests for the complete scene-aware system"""
    
    @pytest.fixture
    async def full_system(self):
        """Create full system with all components"""
        from app.memory.dual_memory_system import DualMemorySystem
        from app.vlm_processors.scene_aware_vlm import SceneAwareVLMProcessor
        from app.vlm_processors.base_processor import VLMConfig
        
        # Create memory system
        memory = DualMemorySystem(mode="dual")
        await memory.initialize()
        
        # Create VLM processor
        config = VLMConfig(use_context=True, context_window=3)
        vlm = SceneAwareVLMProcessor(config)
        vlm.model = Mock()  # Mock to avoid loading
        vlm.processor = Mock()
        vlm.tokenizer = Mock()
        vlm.initialized = True
        
        # Connect VLM to memory
        vlm.set_memory_system(memory)
        
        return {
            'memory': memory,
            'vlm': vlm,
            'generator': MockScreenGenerator()
        }
    
    @pytest.mark.asyncio
    async def test_complete_workflow_no_context_bleeding(self, full_system):
        """Test that context doesn't bleed between different applications"""
        memory = full_system['memory']
        vlm = full_system['vlm']
        generator = full_system['generator']
        
        # Mock VLM responses
        vlm._generate = Mock(side_effect=[
            "User editing README.md file in VSCode",
            "User modifying documentation in VSCode",
            "User browsing Stack Overflow for Python solutions",
            "User reading answers about async programming"
        ])
        
        # Phase 1: Working in VSCode on README
        vscode_frames = []
        for i in range(2):
            frame = generator.generate_screen(ScreenType.CODE_EDITOR)
            metadata = {
                'active_app': 'VSCode',
                'timestamp': time.time() + i,
                'file': 'README.md'
            }
            
            # Process through memory system
            desc = f"VSCode frame {i}"
            await memory.process_frame(frame, desc, metadata)
            
            # Process through VLM
            result = await vlm.process_with_memory(frame, metadata)
            vscode_frames.append(result)
        
        # Verify VSCode context
        assert 'README' in vscode_frames[-1].description
        assert memory.working_memory.current_app == 'VSCode'
        
        # Phase 2: Switch to Chrome (should trigger scene change)
        chrome_frames = []
        for i in range(2):
            frame = generator.generate_screen(ScreenType.BROWSER)
            metadata = {
                'active_app': 'Chrome',
                'timestamp': time.time() + 10 + i,
                'url': 'stackoverflow.com'
            }
            
            # Process through memory system
            desc = f"Chrome frame {i}"
            await memory.process_frame(frame, desc, metadata)
            
            # Process through VLM
            result = await vlm.process_with_memory(frame, metadata)
            chrome_frames.append(result)
        
        # Verify no context bleeding
        assert 'README' not in chrome_frames[-1].description
        assert 'VSCode' not in chrome_frames[-1].description
        assert 'Stack Overflow' in chrome_frames[-1].description
        assert memory.working_memory.current_app == 'Chrome'
        
        # Verify scene change was detected
        stats = memory.get_stats()
        assert stats['scene_changes'] >= 1
    
    @pytest.mark.asyncio
    async def test_gradual_vs_sudden_changes(self, full_system):
        """Test system distinguishes between gradual and sudden changes"""
        memory = full_system['memory']
        vlm = full_system['vlm']
        
        vlm._generate = Mock(return_value="Test description")
        
        # Test 1: Gradual changes (scrolling in same app)
        base_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        gradual_changes = []
        
        for i in range(5):
            # Simulate scrolling by shifting content
            frame = base_frame.copy()
            frame[i*20:(i+1)*20, :] = 255  # Add white stripe
            
            metadata = {'active_app': 'VSCode', 'timestamp': time.time() + i}
            await memory.process_frame(frame, f"Scroll {i}", metadata)
            
            # Check if scene changed
            gradual_changes.append(memory.scene_detector.change_history[-1]['scene_changed'])
            base_frame = frame
        
        # Gradual changes should not trigger scene changes
        assert sum(gradual_changes) <= 1  # Allow max 1 false positive
        
        # Test 2: Sudden change (app switch)
        new_frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        metadata = {'active_app': 'Chrome', 'timestamp': time.time() + 10}
        await memory.process_frame(new_frame, "Browser", metadata)
        
        # Sudden change should trigger scene change
        assert memory.scene_detector.change_history[-1]['scene_changed']
    
    @pytest.mark.asyncio
    async def test_memory_preservation_across_scenes(self, full_system):
        """Test that important information is preserved across scene changes"""
        memory = full_system['memory']
        vlm = full_system['vlm']
        generator = full_system['generator']
        
        vlm._generate = Mock(side_effect=[
            "User fixing critical bug in authentication.py",
            "User testing the authentication fix",
            "User committing changes to git"
        ])
        
        # Work on important bug fix
        for i in range(2):
            frame = generator.generate_screen(ScreenType.CODE_EDITOR)
            metadata = {
                'active_app': 'VSCode',
                'timestamp': time.time() + i,
                'file': 'authentication.py'
            }
            
            desc = f"Bug fix step {i}"
            await memory.process_frame(frame, desc, metadata)
            await vlm.process_with_memory(frame, metadata)
        
        # Add important entities to working memory
        memory.working_memory.entities.add('authentication.py')
        memory.working_memory.entities.add('critical_bug')
        memory.working_memory.current_activity = 'debugging'
        
        # Switch to terminal (scene change)
        terminal_frame = generator.generate_screen(ScreenType.TERMINAL)
        terminal_metadata = {
            'active_app': 'Terminal',
            'timestamp': time.time() + 10
        }
        
        await memory.process_frame(terminal_frame, "Git commit", terminal_metadata)
        
        # Check that facts were saved before clearing
        assert memory.facts_stored > 0
        
        # Query for the bug fix information
        memories = await memory.get_relevant_memories(
            "authentication bug fix",
            {'active_app': 'Terminal'},
            n_results=5
        )
        
        # Should be able to retrieve context about the bug fix
        # (if long-term memory is properly initialized)
        if memories:
            assert any('authentication' in str(m).lower() for m in memories)
    
    @pytest.mark.asyncio
    async def test_rapid_context_switching(self, full_system):
        """Test system handles rapid switching between contexts"""
        memory = full_system['memory']
        vlm = full_system['vlm']
        generator = full_system['generator']
        
        # Simulate rapid app switching
        apps = ['VSCode', 'Chrome', 'Terminal', 'VSCode', 'Chrome']
        screen_types = [
            ScreenType.CODE_EDITOR,
            ScreenType.BROWSER,
            ScreenType.TERMINAL,
            ScreenType.CODE_EDITOR,
            ScreenType.BROWSER
        ]
        
        vlm._generate = Mock(return_value="Activity description")
        
        results = []
        for app, screen_type in zip(apps, screen_types):
            frame = generator.generate_screen(screen_type)
            metadata = {
                'active_app': app,
                'timestamp': time.time()
            }
            
            await memory.process_frame(frame, f"{app} activity", metadata)
            result = await vlm.process_with_memory(frame, metadata)
            results.append({
                'app': app,
                'context_size': result.metadata.get('context_size', 0),
                'scene_id': result.metadata.get('scene_id')
            })
            
            # Small delay to simulate real usage
            await asyncio.sleep(0.1)
        
        # Each app switch should create new scene
        scene_ids = [r['scene_id'] for r in results]
        unique_scenes = len(set(scene_ids))
        assert unique_scenes >= 3  # At least 3 different scenes
        
        # Context should reset on each switch
        for i in range(1, len(results)):
            if results[i]['app'] != results[i-1]['app']:
                # Context should be small after app switch
                assert results[i]['context_size'] <= 1
    
    @pytest.mark.asyncio
    async def test_performance_with_scene_awareness(self, full_system):
        """Test that scene awareness doesn't degrade performance"""
        memory = full_system['memory']
        vlm = full_system['vlm']
        generator = full_system['generator']
        
        vlm._generate = Mock(return_value="Quick description")
        
        # Process 50 frames with scene changes
        start_time = time.time()
        
        for i in range(50):
            # Alternate between apps every 10 frames
            app = 'VSCode' if (i // 10) % 2 == 0 else 'Chrome'
            screen_type = ScreenType.CODE_EDITOR if app == 'VSCode' else ScreenType.BROWSER
            
            frame = generator.generate_screen(screen_type)
            metadata = {
                'active_app': app,
                'timestamp': time.time() + i
            }
            
            await memory.process_frame(frame, f"Frame {i}", metadata)
            await vlm.process_with_memory(frame, metadata)
        
        total_time = time.time() - start_time
        
        # Should process 50 frames in reasonable time (< 5 seconds with mocked model)
        assert total_time < 5.0
        
        # Calculate average processing time
        avg_time = total_time / 50
        assert avg_time < 0.1  # < 100ms per frame average
        
        # Verify scene changes were detected
        stats = memory.get_stats()
        assert stats['scene_changes'] >= 4  # At least 4 app switches
        assert stats['total_processed'] == 50
    
    @pytest.mark.asyncio
    async def test_error_recovery_in_system(self, full_system):
        """Test system recovers from various error conditions"""
        memory = full_system['memory']
        vlm = full_system['vlm']
        generator = full_system['generator']
        
        # Test 1: VLM error
        vlm._generate = Mock(side_effect=Exception("Model error"))
        
        frame = generator.generate_screen(ScreenType.CODE_EDITOR)
        metadata = {'active_app': 'VSCode', 'timestamp': time.time()}
        
        # Should handle VLM error gracefully
        result = await vlm.process_with_memory(frame, metadata)
        assert result.description == "Failed to process image"
        assert result.confidence == 0.0
        
        # Test 2: Memory system error
        memory.long_term_memory = None  # Simulate missing long-term memory
        
        # Should still work with working memory only
        vlm._generate = Mock(return_value="Recovery description")
        result = await vlm.process_with_memory(frame, metadata)
        assert result.description == "Recovery description"
        
        # Test 3: Corrupted frame
        corrupted_frame = None
        result = memory.scene_detector.detect_change(corrupted_frame, metadata)
        assert not result['scene_changed']  # Should handle None frame
        
        # Test 4: Missing metadata
        normal_frame = generator.generate_screen(ScreenType.BROWSER)
        result = memory.scene_detector.detect_change(normal_frame, {})
        assert 'scene_changed' in result  # Should work with empty metadata


class TestEndToEndScenarios:
    """Test real-world usage scenarios"""
    
    @pytest.mark.asyncio
    async def test_developer_workflow(self):
        """Test typical developer workflow with context management"""
        from app.memory.dual_memory_system import DualMemorySystem
        from app.vlm_processors.scene_aware_vlm import SceneAwareVLMProcessor
        from app.vlm_processors.base_processor import VLMConfig
        
        # Setup
        memory = DualMemorySystem(mode="dual")
        await memory.initialize()
        
        config = VLMConfig(use_context=True, context_window=5)
        vlm = SceneAwareVLMProcessor(config)
        vlm.set_memory_system(memory)
        vlm.model = Mock()
        vlm.initialized = True
        
        generator = MockScreenGenerator()
        
        # Mock realistic VLM responses
        vlm._generate = Mock(side_effect=[
            # Coding phase
            "User writing unit tests for authentication module",
            "User implementing test_login_success function",
            "User adding assertion for successful login",
            # Browser phase
            "User searching 'pytest fixtures' on Google",
            "User reading pytest documentation on fixtures",
            # Back to coding
            "User creating pytest fixture for test database",
            "User refactoring tests to use the fixture"
        ])
        
        workflow_results = []
        
        # Phase 1: Writing tests (3 frames)
        for i in range(3):
            frame = generator.generate_screen(ScreenType.CODE_EDITOR)
            metadata = {
                'active_app': 'VSCode',
                'file': 'test_auth.py',
                'timestamp': time.time() + i
            }
            
            await memory.process_frame(frame, f"Testing frame {i}", metadata)
            result = await vlm.process_with_memory(frame, metadata)
            workflow_results.append({
                'phase': 'coding',
                'description': result.description,
                'context_size': result.metadata.get('context_size', 0)
            })
        
        # Verify coding context builds up
        assert workflow_results[-1]['context_size'] > 0
        assert 'test' in workflow_results[-1]['description'].lower()
        
        # Phase 2: Research in browser (2 frames)
        for i in range(2):
            frame = generator.generate_screen(ScreenType.BROWSER)
            metadata = {
                'active_app': 'Chrome',
                'url': 'pytest.org' if i > 0 else 'google.com',
                'timestamp': time.time() + 10 + i
            }
            
            await memory.process_frame(frame, f"Browser frame {i}", metadata)
            result = await vlm.process_with_memory(frame, metadata)
            workflow_results.append({
                'phase': 'research',
                'description': result.description,
                'context_size': result.metadata.get('context_size', 0)
            })
        
        # Verify context switched to browser
        assert 'pytest' in workflow_results[-1]['description'].lower()
        assert 'authentication' not in workflow_results[-1]['description']
        
        # Phase 3: Back to coding with new knowledge (2 frames)
        for i in range(2):
            frame = generator.generate_screen(ScreenType.CODE_EDITOR)
            metadata = {
                'active_app': 'VSCode',
                'file': 'test_auth.py',
                'timestamp': time.time() + 20 + i
            }
            
            await memory.process_frame(frame, f"Coding frame {i+3}", metadata)
            result = await vlm.process_with_memory(frame, metadata)
            workflow_results.append({
                'phase': 'coding_v2',
                'description': result.description,
                'context_size': result.metadata.get('context_size', 0)
            })
        
        # Verify context is fresh for new coding session
        assert 'fixture' in workflow_results[-1]['description'].lower()
        
        # Check overall workflow statistics
        stats = memory.get_stats()
        assert stats['scene_changes'] >= 2  # At least 2 app switches
        assert stats['total_processed'] == 7
        
        # Verify memory preserved important facts
        if memory.facts_stored > 0:
            memories = await memory.get_relevant_memories(
                "pytest testing",
                {'active_app': 'VSCode'},
                n_results=5
            )
            # Should have some relevant memories if long-term storage works
            assert memories is not None


class TestSystemResilience:
    """Test system resilience and edge cases"""
    
    @pytest.mark.asyncio
    async def test_memory_overflow_handling(self):
        """Test system handles memory overflow gracefully"""
        from app.memory.dual_memory_system import DualMemorySystem
        
        memory = DualMemorySystem(
            mode="dual",
            working_memory_size=3  # Small size to test overflow
        )
        await memory.initialize()
        
        generator = MockScreenGenerator()
        
        # Add many frames to test overflow
        for i in range(10):
            frame = generator.generate_screen(ScreenType.CODE_EDITOR)
            metadata = {'active_app': 'VSCode', 'timestamp': time.time() + i}
            await memory.process_frame(frame, f"Frame {i}", metadata)
        
        # Working memory should be limited
        assert len(memory.working_memory.frames) <= 3
        
        # Should keep most recent frames
        last_frame = memory.working_memory.frames[-1]
        assert 'Frame' in last_frame['description']
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test system handles concurrent frame processing"""
        from app.memory.dual_memory_system import DualMemorySystem
        from app.vlm_processors.scene_aware_vlm import SceneAwareVLMProcessor
        
        memory = DualMemorySystem(mode="dual")
        await memory.initialize()
        
        vlm = SceneAwareVLMProcessor()
        vlm.set_memory_system(memory)
        vlm.model = Mock()
        vlm.initialized = True
        vlm._generate = Mock(return_value="Concurrent description")
        
        generator = MockScreenGenerator()
        
        # Create multiple concurrent tasks
        async def process_frame(index):
            frame = generator.generate_screen(ScreenType.CODE_EDITOR)
            metadata = {
                'active_app': 'VSCode',
                'timestamp': time.time() + index
            }
            await memory.process_frame(frame, f"Concurrent {index}", metadata)
            return await vlm.process_with_memory(frame, metadata)
        
        # Process 5 frames concurrently
        tasks = [process_frame(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete without errors
        assert all(not isinstance(r, Exception) for r in results)
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])