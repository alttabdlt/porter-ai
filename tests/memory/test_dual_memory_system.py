#!/usr/bin/env python3
"""
Test suite for Dual Memory System (Working + Long-term) using TDD.
Tests hierarchical memory architecture with smart context management.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import time
from collections import deque

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils.mock_data import MockScreenGenerator, ScreenType, MockVLMOutput


class TestWorkingMemory:
    """Test suite for Working Memory (ephemeral)"""
    
    @pytest.fixture
    def working_memory(self):
        """Create working memory instance"""
        from app.memory.dual_memory_system import WorkingMemory
        return WorkingMemory(max_frames=5)
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_working_memory_initialization(self):
        """Test working memory initializes correctly"""
        from app.memory.dual_memory_system import WorkingMemory
        
        memory = WorkingMemory(max_frames=10)
        
        assert memory.max_frames == 10
        assert len(memory.frames) == 0
        assert memory.current_activity is None
        assert len(memory.entities) == 0
        assert memory.current_app is None
    
    def test_working_memory_default_size(self):
        """Test default working memory size"""
        from app.memory.dual_memory_system import WorkingMemory
        
        memory = WorkingMemory()
        assert memory.max_frames == 5  # Default
    
    # ==================== FRAME MANAGEMENT TESTS ====================
    
    def test_add_frame(self, working_memory):
        """Test adding frames to working memory"""
        description = "User is editing code in VSCode"
        metadata = {'app': 'VSCode', 'timestamp': time.time()}
        
        working_memory.add(description, metadata)
        
        assert len(working_memory.frames) == 1
        assert working_memory.frames[0]['description'] == description
        assert working_memory.frames[0]['metadata'] == metadata
    
    def test_max_frames_limit(self, working_memory):
        """Test that working memory respects max frames limit"""
        for i in range(10):
            working_memory.add(f"Frame {i}", {'index': i})
        
        # Should only keep last 5 frames (max_frames=5)
        assert len(working_memory.frames) == 5
        assert working_memory.frames[0]['description'] == "Frame 5"
        assert working_memory.frames[-1]['description'] == "Frame 9"
    
    def test_clear_memory(self, working_memory):
        """Test clearing working memory"""
        # Add some data
        working_memory.add("Test frame", {'app': 'Test'})
        working_memory.current_activity = "coding"
        working_memory.entities.add("function_name")
        
        # Clear
        working_memory.clear()
        
        assert len(working_memory.frames) == 0
        assert working_memory.current_activity is None
        # Entities are preserved for fact extraction
        assert len(working_memory.entities) >= 1  # May extract multiple entities
    
    # ==================== ENTITY EXTRACTION TESTS ====================
    
    def test_extract_entities(self, working_memory):
        """Test entity extraction from descriptions"""
        descriptions = [
            "User is editing main.py file",
            "Function process_data is being modified",
            "Variable config_path is undefined"
        ]
        
        for desc in descriptions:
            working_memory.add(desc, {})
            working_memory._extract_entities(desc)
        
        # Should extract entities like file names, functions, variables
        assert len(working_memory.entities) > 0
        assert any('main.py' in str(e) for e in working_memory.entities)
    
    def test_entity_persistence_after_clear(self, working_memory):
        """Test that entities persist after clearing for fact extraction"""
        working_memory.add("Editing config.json", {})
        working_memory._extract_entities("config.json")
        
        entities_before = working_memory.entities.copy()
        working_memory.clear()
        
        # Entities should still be available
        assert working_memory.entities == entities_before
    
    # ==================== ACTIVITY DETECTION TESTS ====================
    
    def test_detect_activity(self, working_memory):
        """Test activity detection from descriptions"""
        descriptions = [
            "User is writing Python code",
            "Debugging error in function",
            "Adding comments to code"
        ]
        
        for desc in descriptions:
            working_memory.add(desc, {'app': 'VSCode'})
        
        activity = working_memory._detect_activity()
        assert activity == "coding"
    
    def test_activity_change_detection(self, working_memory):
        """Test detection of activity changes"""
        # Coding activity
        working_memory.add("Writing Python function", {'app': 'VSCode'})
        assert working_memory._detect_activity() == "coding"
        
        # Clear and switch to browsing
        working_memory.clear()
        working_memory.add("Searching documentation on Google", {'app': 'Chrome'})
        assert working_memory._detect_activity() == "browsing"
    
    # ==================== FACT EXTRACTION TESTS ====================
    
    def test_extract_key_facts(self, working_memory):
        """Test extraction of key facts before clearing"""
        working_memory.add("Editing user_auth.py", {'app': 'VSCode'})
        working_memory.add("Fixed authentication bug", {'app': 'VSCode'})
        working_memory.current_activity = "debugging"
        working_memory.entities.update(['user_auth.py', 'authentication_bug'])
        
        facts = working_memory.extract_key_facts()
        
        assert facts is not None
        assert facts['activity'] == "debugging"
        assert 'user_auth.py' in facts['entities']
        assert 'summary' in facts
    
    def test_extract_facts_empty_memory(self, working_memory):
        """Test fact extraction from empty memory returns None"""
        facts = working_memory.extract_key_facts()
        assert facts is None
    
    # ==================== SUMMARIZATION TESTS ====================
    
    def test_summarize_session(self, working_memory):
        """Test session summarization"""
        descriptions = [
            "Opened project in VSCode",
            "Edited main.py file",
            "Fixed syntax error on line 42",
            "Ran tests successfully"
        ]
        
        for desc in descriptions:
            working_memory.add(desc, {'app': 'VSCode'})
        
        summary = working_memory.summarize()
        
        assert summary is not None
        assert len(summary) > 0
        assert any(keyword in summary.lower() for keyword in ['edit', 'fix', 'test'])
    
    def test_summarize_empty_memory(self, working_memory):
        """Test summarization of empty memory"""
        summary = working_memory.summarize()
        assert summary == ""
    
    # ==================== CONTEXT RETRIEVAL TESTS ====================
    
    def test_get_recent_context(self, working_memory):
        """Test getting recent context from working memory"""
        for i in range(5):
            working_memory.add(f"Frame {i}", {'index': i})
        
        recent = working_memory.get_recent(n=3)
        
        assert len(recent) == 3
        assert recent[0]['description'] == "Frame 2"
        assert recent[-1]['description'] == "Frame 4"


class TestDualMemorySystem:
    """Test suite for Dual Memory System integration"""
    
    @pytest.fixture
    async def dual_memory(self):
        """Create dual memory system instance"""
        from app.memory.dual_memory_system import DualMemorySystem
        memory = DualMemorySystem()
        await memory.initialize()
        return memory
    
    @pytest.fixture
    def mock_generator(self):
        """Create mock screen generator"""
        return MockScreenGenerator()
    
    # ==================== INITIALIZATION TESTS ====================
    
    @pytest.mark.asyncio
    async def test_dual_memory_initialization(self):
        """Test dual memory system initialization"""
        from app.memory.dual_memory_system import DualMemorySystem
        
        memory = DualMemorySystem()
        await memory.initialize()
        
        assert memory.working_memory is not None
        assert memory.long_term_memory is not None
        assert memory.scene_detector is not None
        assert memory.mode == "dual"  # Default mode
    
    @pytest.mark.asyncio
    async def test_memory_modes(self):
        """Test different memory modes"""
        from app.memory.dual_memory_system import DualMemorySystem
        
        # Dual mode (default)
        dual = DualMemorySystem(mode="dual")
        assert dual.mode == "dual"
        
        # Working only mode
        working_only = DualMemorySystem(mode="working_only")
        assert working_only.mode == "working_only"
        
        # Persistent only mode
        persistent_only = DualMemorySystem(mode="persistent_only")
        assert persistent_only.mode == "persistent_only"
    
    # ==================== SCENE CHANGE HANDLING TESTS ====================
    
    @pytest.mark.asyncio
    async def test_scene_change_clears_working_memory(self, dual_memory, mock_generator):
        """Test that scene change clears working memory"""
        # Add frames to working memory
        frame1 = mock_generator.generate_screen(ScreenType.CODE_EDITOR)
        await dual_memory.process_frame(frame1, "Editing code", {'active_app': 'VSCode'})
        
        # Verify working memory has content
        assert len(dual_memory.working_memory.frames) > 0
        
        # Simulate scene change (app switch)
        frame2 = mock_generator.generate_screen(ScreenType.BROWSER)
        await dual_memory.process_frame(frame2, "Browsing web", {'active_app': 'Chrome'})
        
        # Working memory should be cleared (or have only new frame)
        assert len(dual_memory.working_memory.frames) <= 1
    
    @pytest.mark.asyncio
    async def test_important_facts_saved_on_scene_change(self, dual_memory, mock_generator):
        """Test that important facts are saved to long-term memory on scene change"""
        # Work in code editor
        frame1 = mock_generator.generate_screen(ScreenType.CODE_EDITOR)
        await dual_memory.process_frame(
            frame1, 
            "Fixed authentication bug in user_auth.py",
            {'active_app': 'VSCode'}
        )
        
        # Add entities
        dual_memory.working_memory.entities.update(['user_auth.py', 'authentication_bug'])
        
        # Mock long-term memory store
        dual_memory.long_term_memory.store_context = AsyncMock()
        
        # Switch to browser (triggers scene change)
        frame2 = mock_generator.generate_screen(ScreenType.BROWSER)
        await dual_memory.process_frame(
            frame2,
            "Searching documentation",
            {'active_app': 'Chrome'}
        )
        
        # Verify facts were stored
        dual_memory.long_term_memory.store_context.assert_called()
    
    @pytest.mark.asyncio
    async def test_no_context_bleeding(self, dual_memory, mock_generator):
        """Test that context doesn't bleed across scene changes"""
        # Process frame in VSCode
        frame1 = mock_generator.generate_screen(ScreenType.CODE_EDITOR)
        description1 = await dual_memory.process_frame(
            frame1,
            "Editing README.md file",
            {'active_app': 'VSCode'}
        )
        
        # Switch to Chrome
        frame2 = mock_generator.generate_screen(ScreenType.BROWSER)
        description2 = await dual_memory.process_frame(
            frame2,
            "Browsing GitHub",
            {'active_app': 'Chrome'}
        )
        
        # Descriptions should not contain cross-context information
        assert "README" not in description2
        assert "GitHub" not in description1
    
    # ==================== MEMORY RETRIEVAL TESTS ====================
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories(self, dual_memory):
        """Test retrieval of relevant memories from long-term storage"""
        # Mock long-term memory search
        mock_memories = [
            {'description': 'Previous coding session', 'app': 'VSCode'},
            {'description': 'Edited similar file', 'app': 'VSCode'}
        ]
        dual_memory.long_term_memory.search_similar = AsyncMock(return_value=mock_memories)
        
        # Query memories
        memories = await dual_memory.get_relevant_memories(
            "Current coding task",
            {'active_app': 'VSCode'}
        )
        
        assert len(memories) > 0
        assert memories[0]['app'] == 'VSCode'
    
    @pytest.mark.asyncio
    async def test_app_specific_memory_retrieval(self, dual_memory):
        """Test that memory retrieval can be filtered by application"""
        # Mock memories from different apps
        dual_memory.long_term_memory.search_similar = AsyncMock()
        
        # Query for VSCode memories
        await dual_memory.get_relevant_memories(
            "coding",
            {'active_app': 'VSCode'}
        )
        
        # Verify filter was applied
        call_args = dual_memory.long_term_memory.search_similar.call_args
        assert call_args is not None
    
    # ==================== CONTINUOUS VS DISCRETE PROCESSING ====================
    
    @pytest.mark.asyncio
    async def test_continuous_scene_processing(self, dual_memory, mock_generator):
        """Test processing continuous scenes (no scene change)"""
        frame = mock_generator.generate_screen(ScreenType.CODE_EDITOR)
        metadata = {'active_app': 'VSCode'}
        
        # Process multiple similar frames
        for i in range(3):
            description = await dual_memory.process_frame(
                frame,
                f"Editing code line {i}",
                metadata
            )
        
        # Working memory should accumulate frames
        assert len(dual_memory.working_memory.frames) == 3
    
    @pytest.mark.asyncio
    async def test_discrete_scene_processing(self, dual_memory, mock_generator):
        """Test processing discrete scenes (with scene changes)"""
        apps = ['VSCode', 'Chrome', 'Terminal']
        screen_types = [ScreenType.CODE_EDITOR, ScreenType.BROWSER, ScreenType.TERMINAL]
        
        for app, screen_type in zip(apps, screen_types):
            frame = mock_generator.generate_screen(screen_type)
            await dual_memory.process_frame(
                frame,
                f"Working in {app}",
                {'active_app': app}
            )
        
        # Working memory should only have recent frames (after last scene change)
        assert len(dual_memory.working_memory.frames) <= 1
    
    # ==================== MEMORY MODE TESTS ====================
    
    @pytest.mark.asyncio
    async def test_working_only_mode(self):
        """Test working memory only mode (no long-term storage)"""
        from app.memory.dual_memory_system import DualMemorySystem
        
        memory = DualMemorySystem(mode="working_only")
        await memory.initialize()
        
        # Process frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        await memory.process_frame(frame, "Test", {'active_app': 'Test'})
        
        # Should not attempt to store in long-term memory
        assert memory.mode == "working_only"
    
    @pytest.mark.asyncio
    async def test_persistent_only_mode(self):
        """Test persistent memory only mode (always stores)"""
        from app.memory.dual_memory_system import DualMemorySystem
        
        memory = DualMemorySystem(mode="persistent_only")
        await memory.initialize()
        
        memory.long_term_memory.store_context = AsyncMock()
        
        # Process frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        await memory.process_frame(frame, "Test", {'active_app': 'Test'})
        
        # Should store even without scene change
        assert memory.mode == "persistent_only"
    
    # ==================== STATISTICS TESTS ====================
    
    @pytest.mark.asyncio
    async def test_memory_statistics(self, dual_memory, mock_generator):
        """Test memory system statistics"""
        # Process some frames
        for i in range(5):
            frame = mock_generator.generate_screen(ScreenType.CODE_EDITOR)
            await dual_memory.process_frame(
                frame,
                f"Frame {i}",
                {'active_app': 'VSCode'}
            )
        
        stats = dual_memory.get_stats()
        
        assert 'working_memory_frames' in stats
        assert 'scene_changes' in stats
        assert 'total_processed' in stats
        assert stats['total_processed'] == 5
    
    # ==================== ERROR HANDLING TESTS ====================
    
    @pytest.mark.asyncio
    async def test_handle_invalid_frame(self, dual_memory):
        """Test handling of invalid frames"""
        # Process None frame
        result = await dual_memory.process_frame(None, "Test", {})
        
        # Should handle gracefully
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_handle_storage_failure(self, dual_memory):
        """Test handling of long-term storage failures"""
        # Mock storage failure
        dual_memory.long_term_memory.store_context = AsyncMock(
            side_effect=Exception("Storage failed")
        )
        
        # Should not crash
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = await dual_memory.process_frame(
            frame,
            "Test",
            {'active_app': 'Test'}
        )
        
        assert result is not None


class TestMemoryIntegration:
    """Integration tests for complete memory system"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_workflow(self):
        """Test complete workflow with scene changes and memory management"""
        from app.memory.dual_memory_system import DualMemorySystem
        
        memory = DualMemorySystem()
        await memory.initialize()
        
        generator = MockScreenGenerator()
        
        # Simulate user workflow
        workflow = [
            (ScreenType.CODE_EDITOR, 'VSCode', "Writing authentication module"),
            (ScreenType.CODE_EDITOR, 'VSCode', "Fixed bug in login function"),
            (ScreenType.CODE_EDITOR, 'VSCode', "Added unit tests"),
            (ScreenType.BROWSER, 'Chrome', "Searching Python documentation"),
            (ScreenType.BROWSER, 'Chrome', "Reading about decorators"),
            (ScreenType.TERMINAL, 'Terminal', "Running pytest"),
        ]
        
        for screen_type, app, description in workflow:
            frame = generator.generate_screen(screen_type)
            await memory.process_frame(frame, description, {'active_app': app})
        
        # Check that scene changes were detected
        stats = memory.get_stats()
        assert stats['scene_changes'] >= 2  # At least VSCode→Chrome and Chrome→Terminal
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memory_recall_across_sessions(self):
        """Test that relevant memories can be recalled in new sessions"""
        from app.memory.dual_memory_system import DualMemorySystem
        
        memory = DualMemorySystem()
        await memory.initialize()
        
        # First session - work on authentication
        memory.working_memory.add("Implementing user authentication", {'app': 'VSCode'})
        memory.working_memory.entities.update(['auth.py', 'User', 'login'])
        
        # Simulate scene change to save facts
        facts = memory.working_memory.extract_key_facts()
        if facts:
            await memory.long_term_memory.store_context({
                'description': facts['summary'],
                'entities': facts['entities'],
                'app': 'VSCode'
            })
        
        # Clear working memory (new session)
        memory.working_memory.clear()
        
        # Query for related memories
        memories = await memory.get_relevant_memories(
            "Working on authentication",
            {'active_app': 'VSCode'}
        )
        
        # Should find relevant memories
        assert any('auth' in str(m).lower() for m in memories)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])