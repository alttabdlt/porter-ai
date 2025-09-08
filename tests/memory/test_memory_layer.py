#!/usr/bin/env python3
"""
Test Memory Layer functionality using TDD.
Tests for ChromaDB integration, context storage, and memory queries.
"""

import unittest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app'))


class TestMemoryLayerSetup(unittest.TestCase):
    """Test basic memory layer setup and initialization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_layer_initialization(self):
        """Test that memory layer can be initialized"""
        from memory.memory_layer import MemoryLayer
        
        # Initialize memory layer with temp directory
        memory = MemoryLayer(persist_directory=self.temp_dir)
        
        # Assert initialization successful
        self.assertIsNotNone(memory)
        self.assertEqual(memory.persist_directory, self.temp_dir)
        self.assertIsNotNone(memory.client)
        self.assertIsNotNone(memory.collection)
    
    def test_chromadb_collection_creation(self):
        """Test that ChromaDB collection is created properly"""
        from memory.memory_layer import MemoryLayer
        
        # Initialize memory layer
        memory = MemoryLayer(
            persist_directory=self.temp_dir,
            collection_name="test_contexts"
        )
        
        # Check collection exists
        self.assertEqual(memory.collection.name, "test_contexts")
        self.assertIsNotNone(memory.collection.id)
    
    def test_embedding_model_initialization(self):
        """Test that embedding model is properly initialized"""
        from memory.memory_layer import MemoryLayer
        
        memory = MemoryLayer(persist_directory=self.temp_dir)
        
        # Test embedding generation
        test_text = "This is a test context"
        embedding = memory.generate_embedding(test_text)
        
        # Check embedding properties
        self.assertIsInstance(embedding, list)
        self.assertTrue(len(embedding) > 0)
        self.assertIsInstance(embedding[0], float)


class TestContextStorage(unittest.TestCase):
    """Test storing and retrieving context from memory"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        from memory.memory_layer import MemoryLayer
        self.memory = MemoryLayer(persist_directory=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_store_context(self):
        """Test storing a context in memory"""
        # Create test context
        context = {
            'timestamp': datetime.now().isoformat(),
            'description': 'User is coding in VS Code',
            'activity': 'coding',
            'application': 'Visual Studio Code',
            'importance': 0.8,
            'frame_id': 'frame_001'
        }
        
        # Store context
        context_id = await self.memory.store_context(context)
        
        # Verify stored
        self.assertIsNotNone(context_id)
        self.assertIsInstance(context_id, str)
    
    async def test_store_multiple_contexts(self):
        """Test storing multiple contexts"""
        contexts = [
            {
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                'description': 'User opened Chrome browser',
                'activity': 'browsing',
                'application': 'Google Chrome',
                'importance': 0.5
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=3)).isoformat(),
                'description': 'User is watching YouTube video',
                'activity': 'watching',
                'application': 'Google Chrome',
                'importance': 0.6
            },
            {
                'timestamp': datetime.now().isoformat(),
                'description': 'User switched to Slack',
                'activity': 'messaging',
                'application': 'Slack',
                'importance': 0.7
            }
        ]
        
        # Store all contexts
        context_ids = []
        for context in contexts:
            context_id = await self.memory.store_context(context)
            context_ids.append(context_id)
        
        # Verify all stored
        self.assertEqual(len(context_ids), 3)
        self.assertEqual(len(set(context_ids)), 3)  # All unique IDs
    
    async def test_context_with_screenshot(self):
        """Test storing context with associated screenshot"""
        # Create mock screenshot
        screenshot = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'description': 'User is in Photoshop editing an image',
            'activity': 'design',
            'application': 'Adobe Photoshop',
            'importance': 0.9,
            'screenshot_shape': screenshot.shape
        }
        
        # Store context with screenshot reference
        context_id = await self.memory.store_context(context, screenshot=screenshot)
        
        # Verify stored with screenshot metadata
        self.assertIsNotNone(context_id)
        retrieved = await self.memory.get_context(context_id)
        self.assertEqual(retrieved['screenshot_shape'], list(screenshot.shape))


class TestMemoryQueries(unittest.TestCase):
    """Test querying and searching memory"""
    
    def setUp(self):
        """Set up test fixtures with pre-populated memory"""
        self.temp_dir = tempfile.mkdtemp()
        from memory.memory_layer import MemoryLayer
        self.memory = MemoryLayer(persist_directory=self.temp_dir)
        
        # Pre-populate with test data
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._populate_test_data()
        
    def tearDown(self):
        """Clean up test fixtures"""
        self.loop.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _populate_test_data(self):
        """Populate memory with test contexts"""
        contexts = [
            {
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'description': 'User opened VS Code and started Python project',
                'activity': 'coding',
                'application': 'VS Code',
                'importance': 0.8
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=1, minutes=30)).isoformat(),
                'description': 'User debugged Python code with breakpoints',
                'activity': 'debugging',
                'application': 'VS Code',
                'importance': 0.9
            },
            {
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                'description': 'User searched Stack Overflow for Python errors',
                'activity': 'research',
                'application': 'Chrome',
                'importance': 0.7
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat(),
                'description': 'User wrote unit tests in pytest',
                'activity': 'testing',
                'application': 'VS Code',
                'importance': 0.85
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=10)).isoformat(),
                'description': 'User committed code to Git repository',
                'activity': 'version control',
                'application': 'Terminal',
                'importance': 0.6
            }
        ]
        
        for context in contexts:
            self.loop.run_until_complete(self.memory.store_context(context))
    
    async def test_semantic_search(self):
        """Test semantic search for similar contexts"""
        # Search for Python-related activities
        results = await self.memory.search_similar(
            query="Python programming",
            n_results=3
        )
        
        # Should return relevant Python contexts
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        
        # Check relevance
        for result in results:
            self.assertIn('python', result['description'].lower())
    
    async def test_time_range_query(self):
        """Test querying contexts within time range"""
        # Query last hour
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        results = await self.memory.query_time_range(
            start_time=start_time,
            end_time=end_time
        )
        
        # Should return contexts from last hour
        self.assertGreater(len(results), 0)
        
        # Verify timestamps in range
        for result in results:
            timestamp = datetime.fromisoformat(result['timestamp'])
            self.assertGreaterEqual(timestamp, start_time)
            self.assertLessEqual(timestamp, end_time)
    
    async def test_application_filter(self):
        """Test filtering contexts by application"""
        # Query VS Code activities
        results = await self.memory.query_by_application('VS Code')
        
        # Should return VS Code contexts
        self.assertGreater(len(results), 0)
        
        # Verify all from VS Code
        for result in results:
            self.assertEqual(result['application'], 'VS Code')
    
    async def test_importance_threshold(self):
        """Test querying high-importance contexts"""
        # Query important contexts (>= 0.8)
        results = await self.memory.query_important(threshold=0.8)
        
        # Should return high importance contexts
        self.assertGreater(len(results), 0)
        
        # Verify importance levels
        for result in results:
            self.assertGreaterEqual(result['importance'], 0.8)
    
    async def test_activity_summary(self):
        """Test generating activity summary"""
        # Get activity summary for last 2 hours
        summary = await self.memory.get_activity_summary(
            hours=2
        )
        
        # Should have activity counts
        self.assertIn('total_contexts', summary)
        self.assertIn('activities', summary)
        self.assertIn('applications', summary)
        self.assertIn('average_importance', summary)
        
        # Verify counts
        self.assertGreater(summary['total_contexts'], 0)
        self.assertIsInstance(summary['activities'], dict)
        self.assertIsInstance(summary['applications'], dict)


class TestMemoryPersistence(unittest.TestCase):
    """Test memory persistence across sessions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_persistence_across_sessions(self):
        """Test that memory persists when reloading"""
        from memory.memory_layer import MemoryLayer
        
        # First session - store context
        memory1 = MemoryLayer(persist_directory=self.temp_dir)
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'description': 'Test persistence context',
            'activity': 'testing',
            'importance': 0.5
        }
        
        context_id = await memory1.store_context(context)
        
        # Close first session
        del memory1
        
        # Second session - retrieve context
        memory2 = MemoryLayer(persist_directory=self.temp_dir)
        
        # Should be able to retrieve
        retrieved = await memory2.get_context(context_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['description'], 'Test persistence context')
    
    async def test_memory_size_management(self):
        """Test memory size limits and cleanup"""
        from memory.memory_layer import MemoryLayer
        
        # Initialize with size limit
        memory = MemoryLayer(
            persist_directory=self.temp_dir,
            max_contexts=100
        )
        
        # Store many contexts
        for i in range(150):
            context = {
                'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
                'description': f'Context {i}',
                'importance': 0.5
            }
            await memory.store_context(context)
        
        # Should maintain size limit
        count = await memory.get_context_count()
        self.assertLessEqual(count, 100)
        
        # Should keep most recent contexts
        results = await memory.query_time_range(
            start_time=datetime.now() - timedelta(minutes=10),
            end_time=datetime.now()
        )
        self.assertGreater(len(results), 0)


class TestMemoryIntegration(unittest.TestCase):
    """Test integration with streaming pipeline"""
    
    @patch('memory.memory_layer.chromadb')
    async def test_pipeline_integration(self, mock_chromadb):
        """Test memory layer integration with streaming pipeline"""
        from memory.memory_layer import MemoryLayer
        from streaming.context_fusion import Context
        
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        memory = MemoryLayer()
        
        # Create context from pipeline
        context = Context(
            timestamp=datetime.now().timestamp(),
            vlm_output="User is coding in Python",
            importance=0.8
        )
        
        # Store pipeline context
        context_dict = context.to_dict()
        context_id = await memory.store_context(context_dict)
        
        # Verify stored
        self.assertIsNotNone(context_id)
        mock_collection.add.assert_called_once()
    
    async def test_memory_based_context_enhancement(self):
        """Test using memory to enhance current context"""
        from memory.memory_layer import MemoryLayer
        
        memory = MemoryLayer(persist_directory=self.temp_dir)
        
        # Store historical context
        past_context = {
            'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
            'description': 'User was editing config.py file',
            'activity': 'coding',
            'file': 'config.py',
            'importance': 0.7
        }
        await memory.store_context(past_context)
        
        # Current context
        current_description = "User is still in VS Code"
        
        # Enhance with memory
        enhanced = await memory.enhance_with_memory(
            current_description,
            lookback_minutes=10
        )
        
        # Should include historical context
        self.assertIn('recent_contexts', enhanced)
        self.assertGreater(len(enhanced['recent_contexts']), 0)
        self.assertIn('suggested_context', enhanced)


def run_tests():
    """Run all memory layer tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryLayerSetup))
    suite.addTests(loader.loadTestsFromTestCase(TestContextStorage))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryQueries))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryPersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_async_test(test_func):
    """Helper to run async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_func())
    finally:
        loop.close()


if __name__ == '__main__':
    # Run standard tests
    success = run_tests()
    
    # Run async tests
    test_storage = TestContextStorage()
    test_queries = TestMemoryQueries()
    test_persistence = TestMemoryPersistence()
    test_integration = TestMemoryIntegration()
    
    print("\n--- Running Async Tests ---")
    
    async_tests = [
        # Context Storage tests
        test_storage.test_store_context,
        test_storage.test_store_multiple_contexts,
        test_storage.test_context_with_screenshot,
        
        # Query tests
        test_queries.test_semantic_search,
        test_queries.test_time_range_query,
        test_queries.test_application_filter,
        test_queries.test_importance_threshold,
        test_queries.test_activity_summary,
        
        # Persistence tests
        test_persistence.test_persistence_across_sessions,
        test_persistence.test_memory_size_management,
        
        # Integration tests
        test_integration.test_pipeline_integration,
        test_integration.test_memory_based_context_enhancement
    ]
    
    for test in async_tests:
        try:
            print(f"Running {test.__name__}...")
            # Set up fixtures for each test
            if hasattr(test.__self__, 'setUp'):
                test.__self__.setUp()
            
            run_async_test(test)
            print(f"✓ {test.__name__} passed")
            
            # Tear down fixtures
            if hasattr(test.__self__, 'tearDown'):
                test.__self__.tearDown()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            success = False
    
    sys.exit(0 if success else 1)