#!/usr/bin/env python3
"""
Test enhanced Memory Layer functionality with better embeddings.
Tests for sentence-transformers, temporal weighting, and richer metadata.
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
import time

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app'))


class TestEnhancedEmbeddings(unittest.TestCase):
    """Test improved embedding generation for better semantic search"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sentence_transformer_initialization(self):
        """Test that sentence-transformers can be initialized"""
        from memory.enhanced_memory import EnhancedMemoryLayer
        
        memory = EnhancedMemoryLayer(persist_directory=self.temp_dir)
        
        # Should use sentence-transformers
        self.assertIsNotNone(memory.sentence_model)
        self.assertEqual(memory.embedding_model_name, 'all-MiniLM-L6-v2')
        
    def test_embedding_quality_comparison(self):
        """Test that sentence-transformers provide better embeddings than default"""
        from memory.enhanced_memory import EnhancedMemoryLayer
        from memory.memory_layer import MemoryLayer
        
        enhanced_memory = EnhancedMemoryLayer(persist_directory=self.temp_dir)
        default_memory = MemoryLayer(persist_directory=self.temp_dir + "_default")
        
        # Test texts with known semantic similarity
        text1 = "User is writing Python code in VS Code"
        text2 = "User is programming in Python using Visual Studio Code"
        text3 = "User is watching YouTube videos"
        
        # Get embeddings from both
        enhanced_emb1 = enhanced_memory.generate_embedding(text1)
        enhanced_emb2 = enhanced_memory.generate_embedding(text2)
        enhanced_emb3 = enhanced_memory.generate_embedding(text3)
        
        default_emb1 = default_memory.generate_embedding(text1)
        default_emb2 = default_memory.generate_embedding(text2)
        default_emb3 = default_memory.generate_embedding(text3)
        
        # Calculate cosine similarities
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Enhanced should show higher similarity for semantically similar texts
        enhanced_sim_similar = cosine_sim(enhanced_emb1, enhanced_emb2)
        enhanced_sim_different = cosine_sim(enhanced_emb1, enhanced_emb3)
        
        default_sim_similar = cosine_sim(default_emb1, default_emb2)
        default_sim_different = cosine_sim(default_emb1, default_emb3)
        
        # Enhanced embeddings should better distinguish similar vs different
        enhanced_ratio = enhanced_sim_similar / max(enhanced_sim_different, 0.01)
        default_ratio = default_sim_similar / max(default_sim_different, 0.01)
        
        # Enhanced should have better discrimination (higher ratio)
        self.assertGreater(enhanced_ratio, default_ratio * 1.2)  # 20% better
        
    async def test_temporal_weighting(self):
        """Test that recent contexts are weighted higher in search"""
        from memory.enhanced_memory import EnhancedMemoryLayer
        
        memory = EnhancedMemoryLayer(persist_directory=self.temp_dir)
        
        # Store contexts at different times
        old_context = {
            'timestamp': (datetime.now() - timedelta(hours=24)).isoformat(),
            'description': 'User is coding in Python',
            'importance': 0.5
        }
        
        recent_context = {
            'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
            'description': 'User is coding in Python',
            'importance': 0.5
        }
        
        old_id = await memory.store_context(old_context)
        recent_id = await memory.store_context(recent_context)
        
        # Search for similar contexts
        results = await memory.search_similar_weighted(
            query="Python programming",
            n_results=2,
            use_temporal_weight=True
        )
        
        # Recent context should rank higher despite same content
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], recent_id)
        self.assertGreater(results[0]['weighted_score'], results[1]['weighted_score'])
    
    def test_embedding_dimension_consistency(self):
        """Test that embeddings have consistent dimensions"""
        from memory.enhanced_memory import EnhancedMemoryLayer
        
        memory = EnhancedMemoryLayer(persist_directory=self.temp_dir)
        
        texts = [
            "Short text",
            "A much longer text with many words that should still produce the same dimension embedding",
            "中文文本测试",  # Test with non-English
            ""  # Empty text
        ]
        
        embeddings = [memory.generate_embedding(text) for text in texts]
        
        # All should have same dimensions
        dimensions = [len(emb) for emb in embeddings if emb]
        self.assertTrue(all(d == dimensions[0] for d in dimensions))
        self.assertEqual(dimensions[0], 384)  # all-MiniLM-L6-v2 dimension


class TestEnhancedMetadata(unittest.TestCase):
    """Test richer context metadata storage"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        from memory.enhanced_memory import EnhancedMemoryLayer
        self.memory = EnhancedMemoryLayer(persist_directory=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_window_title_storage(self):
        """Test that window titles are captured and stored"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'description': 'User is coding',
            'window_title': 'main.py - Visual Studio Code',
            'application': 'VS Code',
            'importance': 0.8
        }
        
        context_id = await self.memory.store_enhanced_context(context)
        retrieved = await self.memory.get_context(context_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['window_title'], 'main.py - Visual Studio Code')
        self.assertIn('file_name', retrieved)
        self.assertEqual(retrieved['file_name'], 'main.py')
    
    async def test_url_extraction_storage(self):
        """Test that URLs are extracted from browser contexts"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'description': 'User is browsing Stack Overflow',
            'window_title': 'python - How to use async/await - Stack Overflow',
            'url': 'https://stackoverflow.com/questions/123456',
            'application': 'Chrome',
            'importance': 0.6
        }
        
        context_id = await self.memory.store_enhanced_context(context)
        retrieved = await self.memory.get_context(context_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['url'], 'https://stackoverflow.com/questions/123456')
        self.assertIn('domain', retrieved)
        self.assertEqual(retrieved['domain'], 'stackoverflow.com')
    
    async def test_application_state_capture(self):
        """Test that application state is properly captured"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'description': 'User is debugging Python code',
            'window_title': 'app.py - Visual Studio Code',
            'application': 'VS Code',
            'app_state': {
                'file': 'app.py',
                'language': 'python',
                'cursor_line': 42,
                'has_breakpoint': True,
                'terminal_open': True
            },
            'importance': 0.9
        }
        
        context_id = await self.memory.store_enhanced_context(context)
        retrieved = await self.memory.get_context(context_id)
        
        self.assertIsNotNone(retrieved)
        self.assertIn('app_state', retrieved)
        self.assertEqual(retrieved['app_state']['cursor_line'], 42)
        self.assertTrue(retrieved['app_state']['has_breakpoint'])
    
    async def test_project_linking(self):
        """Test that related contexts are linked by project"""
        contexts = [
            {
                'timestamp': (datetime.now() - timedelta(minutes=10)).isoformat(),
                'description': 'User opened project',
                'window_title': 'porter.ai - Visual Studio Code',
                'project': 'porter.ai',
                'importance': 0.5
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                'description': 'User editing main file',
                'window_title': 'main.py - porter.ai - Visual Studio Code',
                'project': 'porter.ai',
                'importance': 0.7
            },
            {
                'timestamp': datetime.now().isoformat(),
                'description': 'User running tests',
                'window_title': 'Terminal - porter.ai',
                'project': 'porter.ai',
                'importance': 0.8
            }
        ]
        
        context_ids = []
        for ctx in contexts:
            ctx_id = await self.memory.store_enhanced_context(ctx)
            context_ids.append(ctx_id)
        
        # Query by project
        project_contexts = await self.memory.query_by_project('porter.ai')
        
        self.assertEqual(len(project_contexts), 3)
        self.assertTrue(all(c['project'] == 'porter.ai' for c in project_contexts))
        
        # Should be ordered by timestamp
        timestamps = [c['timestamp'] for c in project_contexts]
        self.assertEqual(timestamps, sorted(timestamps))


class TestSmarterQueries(unittest.TestCase):
    """Test improved query capabilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        from memory.enhanced_memory import EnhancedMemoryLayer
        self.memory = EnhancedMemoryLayer(persist_directory=self.temp_dir)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        """Clean up test fixtures"""
        self.loop.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_recency_weighted_search(self):
        """Test that search can be weighted by recency"""
        # Store contexts over time
        for i in range(5):
            context = {
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'description': f'User is coding iteration {i}',
                'importance': 0.5
            }
            await self.memory.store_context(context)
        
        # Search with recency weighting
        results = await self.memory.search_similar_weighted(
            query="coding",
            n_results=3,
            recency_weight=0.7  # 70% weight on recency
        )
        
        # Should return most recent first
        self.assertEqual(len(results), 3)
        
        # Check that results are ordered by recency-weighted score
        for i in range(len(results) - 1):
            self.assertGreater(
                results[i]['weighted_score'],
                results[i + 1]['weighted_score']
            )
    
    async def test_activity_clustering(self):
        """Test that similar activities are clustered"""
        # Store various coding activities
        coding_contexts = [
            "User is writing Python code",
            "User is debugging Python script",
            "User is refactoring functions",
            "User is writing unit tests",
            "User is reviewing code"
        ]
        
        for desc in coding_contexts:
            await self.memory.store_context({
                'timestamp': datetime.now().isoformat(),
                'description': desc,
                'activity': 'coding',
                'importance': 0.7
            })
        
        # Store other activities
        other_contexts = [
            "User is checking email",
            "User is in Zoom meeting",
            "User is browsing news"
        ]
        
        for desc in other_contexts:
            await self.memory.store_context({
                'timestamp': datetime.now().isoformat(),
                'description': desc,
                'activity': 'other',
                'importance': 0.5
            })
        
        # Get activity clusters
        clusters = await self.memory.get_activity_clusters()
        
        self.assertIn('coding', clusters)
        self.assertEqual(len(clusters['coding']), 5)
        self.assertIn('other', clusters)
        self.assertEqual(len(clusters['other']), 3)
    
    async def test_pattern_extraction(self):
        """Test extraction of repeated workflow patterns"""
        # Simulate a repeated workflow pattern
        workflow_pattern = [
            ("User opens VS Code", "coding"),
            ("User opens terminal", "coding"),
            ("User runs tests", "testing"),
            ("User fixes code", "debugging"),
            ("User runs tests", "testing"),
            ("User commits changes", "version_control")
        ]
        
        # Repeat pattern 3 times
        for _ in range(3):
            for desc, activity in workflow_pattern:
                await self.memory.store_context({
                    'timestamp': datetime.now().isoformat(),
                    'description': desc,
                    'activity': activity,
                    'importance': 0.6
                })
                await asyncio.sleep(0.01)  # Small delay to ensure ordering
        
        # Extract patterns
        patterns = await self.memory.extract_workflow_patterns(
            min_occurrences=2,
            window_size=6
        )
        
        self.assertGreater(len(patterns), 0)
        
        # Should detect the coding->testing->debugging pattern
        top_pattern = patterns[0]
        self.assertIn('coding', top_pattern['activities'])
        self.assertIn('testing', top_pattern['activities'])
        self.assertIn('debugging', top_pattern['activities'])
        self.assertGreaterEqual(top_pattern['occurrences'], 2)
    
    async def test_semantic_activity_grouping(self):
        """Test that semantically similar activities are grouped"""
        activities = [
            ("User is writing Python code", "development"),
            ("User is coding in JavaScript", "development"),
            ("User is debugging errors", "development"),
            ("User is reading documentation", "learning"),
            ("User is watching tutorial video", "learning"),
            ("User is in Slack", "communication"),
            ("User is writing email", "communication")
        ]
        
        for desc, expected_group in activities:
            await self.memory.store_context({
                'timestamp': datetime.now().isoformat(),
                'description': desc,
                'importance': 0.5
            })
        
        # Get semantic groups
        groups = await self.memory.get_semantic_activity_groups()
        
        # Should group by semantic similarity
        self.assertIn('development', groups)
        self.assertIn('learning', groups)
        self.assertIn('communication', groups)
        
        # Each group should have related activities
        self.assertGreaterEqual(len(groups['development']), 3)
        self.assertGreaterEqual(len(groups['learning']), 2)
        self.assertGreaterEqual(len(groups['communication']), 2)


class TestMemoryPerformance(unittest.TestCase):
    """Test performance improvements in memory operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_search_performance(self):
        """Test that enhanced search is still performant"""
        from memory.enhanced_memory import EnhancedMemoryLayer
        
        memory = EnhancedMemoryLayer(persist_directory=self.temp_dir)
        
        # Store 100 contexts
        for i in range(100):
            context = {
                'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
                'description': f'User is performing task {i % 10}',
                'importance': 0.5
            }
            await memory.store_context(context)
        
        # Measure search time
        start_time = time.time()
        results = await memory.search_similar_weighted(
            query="performing task",
            n_results=10,
            use_temporal_weight=True
        )
        search_time = time.time() - start_time
        
        # Should be fast even with enhanced features
        self.assertLess(search_time, 0.5)  # Less than 500ms
        self.assertEqual(len(results), 10)
    
    async def test_batch_storage_performance(self):
        """Test batch storage of contexts"""
        from memory.enhanced_memory import EnhancedMemoryLayer
        
        memory = EnhancedMemoryLayer(persist_directory=self.temp_dir)
        
        # Prepare batch of contexts
        contexts = []
        for i in range(50):
            contexts.append({
                'timestamp': datetime.now().isoformat(),
                'description': f'Batch context {i}',
                'window_title': f'Window {i}',
                'importance': 0.5
            })
        
        # Measure batch storage time
        start_time = time.time()
        ids = await memory.store_batch_contexts(contexts)
        batch_time = time.time() - start_time
        
        # Should handle batch efficiently
        self.assertLess(batch_time, 2.0)  # Less than 2 seconds for 50
        self.assertEqual(len(ids), 50)


def run_tests():
    """Run all enhanced memory tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedMetadata))
    suite.addTests(loader.loadTestsFromTestCase(TestSmarterQueries))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryPerformance))
    
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
    test_cases = [
        TestEnhancedEmbeddings(),
        TestEnhancedMetadata(),
        TestSmarterQueries(),
        TestMemoryPerformance()
    ]
    
    print("\n--- Running Async Tests ---")
    
    for test_case in test_cases:
        # Get all test methods
        test_methods = [method for method in dir(test_case) 
                       if method.startswith('test_') and callable(getattr(test_case, method))]
        
        for method_name in test_methods:
            test_method = getattr(test_case, method_name)
            if asyncio.iscoroutinefunction(test_method):
                try:
                    print(f"Running {test_case.__class__.__name__}.{method_name}...")
                    if hasattr(test_case, 'setUp'):
                        test_case.setUp()
                    
                    run_async_test(test_method)
                    print(f"✓ {method_name} passed")
                    
                    if hasattr(test_case, 'tearDown'):
                        test_case.tearDown()
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    success = False
    
    sys.exit(0 if success else 1)