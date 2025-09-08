#!/usr/bin/env python3
"""
Test Intent Router functionality using TDD.
Tests for intent classification, policy-based routing, and confidence scoring.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path
from datetime import datetime
import json

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app'))


class TestIntentClassification(unittest.TestCase):
    """Test basic intent classification from descriptions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from intent.intent_router import IntentRouter
        self.router = IntentRouter()
    
    async def test_classify_coding_intent(self):
        """Test classification of coding-related intents"""
        descriptions = [
            "User is writing Python code in VS Code",
            "User is debugging JavaScript with breakpoints",
            "User is reviewing pull request on GitHub",
            "User is writing unit tests in pytest"
        ]
        
        for description in descriptions:
            intent = await self.router.classify_intent(description)
            
            self.assertIsNotNone(intent)
            self.assertEqual(intent.primary_intent, 'coding')
            self.assertGreater(intent.confidence, 0.7)
    
    async def test_classify_browsing_intent(self):
        """Test classification of browsing intents"""
        descriptions = [
            "User is browsing Stack Overflow for solutions",
            "User is reading documentation on MDN",
            "User is searching Google for Python tutorials",
            "User is watching YouTube video about React"
        ]
        
        for description in descriptions:
            intent = await self.router.classify_intent(description)
            
            self.assertIsNotNone(intent)
            self.assertIn(intent.primary_intent, ['browsing', 'learning'])
            self.assertGreater(intent.confidence, 0.6)
    
    async def test_classify_communication_intent(self):
        """Test classification of communication intents"""
        descriptions = [
            "User is typing message in Slack",
            "User is composing email in Gmail",
            "User is in video call on Zoom",
            "User is chatting on Discord"
        ]
        
        for description in descriptions:
            intent = await self.router.classify_intent(description)
            
            self.assertIsNotNone(intent)
            self.assertEqual(intent.primary_intent, 'communication')
            self.assertGreater(intent.confidence, 0.7)
    
    async def test_classify_productivity_intent(self):
        """Test classification of productivity intents"""
        descriptions = [
            "User is editing document in Google Docs",
            "User is creating presentation in PowerPoint",
            "User is updating spreadsheet in Excel",
            "User is organizing tasks in Notion"
        ]
        
        for description in descriptions:
            intent = await self.router.classify_intent(description)
            
            self.assertIsNotNone(intent)
            self.assertEqual(intent.primary_intent, 'productivity')
            self.assertGreater(intent.confidence, 0.6)
    
    async def test_multi_intent_detection(self):
        """Test detection of multiple intents in one context"""
        description = "User is coding in VS Code while referencing Stack Overflow documentation"
        
        intent = await self.router.classify_intent(description)
        
        self.assertIsNotNone(intent)
        self.assertEqual(intent.primary_intent, 'coding')
        self.assertIn('learning', intent.secondary_intents)
        self.assertGreater(len(intent.all_intents), 1)
    
    async def test_unknown_intent_handling(self):
        """Test handling of unclear or unknown intents"""
        descriptions = [
            "User is looking at desktop",
            "Screen shows various windows",
            ""
        ]
        
        for description in descriptions:
            intent = await self.router.classify_intent(description)
            
            self.assertIsNotNone(intent)
            self.assertEqual(intent.primary_intent, 'unknown')
            self.assertLess(intent.confidence, 0.5)


class TestPolicyBasedRouting(unittest.TestCase):
    """Test policy-based routing decisions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from intent.intent_router import IntentRouter, RoutingPolicy
        
        # Create router with custom policies
        self.policies = [
            RoutingPolicy(
                name="work_hours_focus",
                condition=lambda ctx: 9 <= ctx['hour'] <= 17,
                allowed_intents=['coding', 'productivity'],
                blocked_intents=['entertainment'],
                priority=10
            ),
            RoutingPolicy(
                name="break_time",
                condition=lambda ctx: ctx['hour'] in [12, 15],
                allowed_intents=['browsing', 'entertainment'],
                priority=5
            ),
            RoutingPolicy(
                name="meeting_mode",
                condition=lambda ctx: ctx.get('in_meeting', False),
                allowed_intents=['communication'],
                blocked_intents=['coding', 'browsing'],
                priority=20
            )
        ]
        
        self.router = IntentRouter(policies=self.policies)
    
    async def test_work_hours_policy(self):
        """Test work hours focus policy"""
        context = {
            'hour': 10,
            'description': 'User is browsing Reddit'
        }
        
        intent = await self.router.classify_intent(context['description'])
        decision = await self.router.route_intent(intent, context)
        
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, 'block')
        self.assertEqual(decision.reason, 'work_hours_focus')
        self.assertIn('entertainment', decision.blocked_intents)
    
    async def test_break_time_policy(self):
        """Test break time allowance"""
        context = {
            'hour': 12,
            'description': 'User is watching YouTube'
        }
        
        intent = await self.router.classify_intent(context['description'])
        decision = await self.router.route_intent(intent, context)
        
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, 'allow')
        self.assertEqual(decision.reason, 'break_time')
    
    async def test_meeting_mode_policy(self):
        """Test meeting mode restrictions"""
        context = {
            'hour': 14,
            'in_meeting': True,
            'description': 'User is coding in VS Code'
        }
        
        intent = await self.router.classify_intent(context['description'])
        decision = await self.router.route_intent(intent, context)
        
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, 'block')
        self.assertEqual(decision.reason, 'meeting_mode')
        self.assertIn('coding', decision.blocked_intents)
    
    async def test_policy_priority(self):
        """Test that higher priority policies override lower ones"""
        context = {
            'hour': 12,  # Break time
            'in_meeting': True,  # But in meeting (higher priority)
            'description': 'User is browsing news'
        }
        
        intent = await self.router.classify_intent(context['description'])
        decision = await self.router.route_intent(intent, context)
        
        # Meeting mode (priority 20) should override break time (priority 5)
        self.assertEqual(decision.action, 'block')
        self.assertEqual(decision.reason, 'meeting_mode')
    
    async def test_no_applicable_policy(self):
        """Test behavior when no policy applies"""
        context = {
            'hour': 18,  # After work
            'description': 'User is playing a game'
        }
        
        intent = await self.router.classify_intent(context['description'])
        decision = await self.router.route_intent(intent, context)
        
        self.assertEqual(decision.action, 'allow')
        self.assertEqual(decision.reason, 'default')


class TestConfidenceScoring(unittest.TestCase):
    """Test intent confidence scoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        from intent.intent_router import IntentRouter
        self.router = IntentRouter()
    
    async def test_high_confidence_clear_intent(self):
        """Test high confidence for clear intents"""
        clear_descriptions = [
            "User is writing Python code with syntax highlighting in VS Code",
            "User is in Zoom video call with 5 participants",
            "User is editing PowerPoint presentation with charts"
        ]
        
        for description in clear_descriptions:
            intent = await self.router.classify_intent(description)
            self.assertGreater(intent.confidence, 0.8)
    
    async def test_medium_confidence_mixed_intent(self):
        """Test medium confidence for mixed intents"""
        mixed_descriptions = [
            "User has multiple windows open",
            "User is switching between applications",
            "User is looking at code and documentation"
        ]
        
        for description in mixed_descriptions:
            intent = await self.router.classify_intent(description)
            self.assertGreater(intent.confidence, 0.4)
            self.assertLess(intent.confidence, 0.8)
    
    async def test_low_confidence_vague_intent(self):
        """Test low confidence for vague descriptions"""
        vague_descriptions = [
            "User is on computer",
            "Screen shows content",
            "Activity detected"
        ]
        
        for description in vague_descriptions:
            intent = await self.router.classify_intent(description)
            self.assertLess(intent.confidence, 0.4)
    
    async def test_confidence_with_keywords(self):
        """Test that specific keywords increase confidence"""
        # Without keywords
        intent1 = await self.router.classify_intent("User is working")
        
        # With specific keywords
        intent2 = await self.router.classify_intent("User is debugging Python code")
        
        self.assertGreater(intent2.confidence, intent1.confidence)
    
    async def test_confidence_decay_over_time(self):
        """Test that confidence decays for stale classifications"""
        description = "User is coding in VS Code"
        
        # Fresh classification
        intent1 = await self.router.classify_intent(description)
        initial_confidence = intent1.confidence
        
        # Simulate time passing (5 minutes)
        intent1.age_minutes = 5
        decayed_confidence = self.router.get_decayed_confidence(intent1)
        
        self.assertLess(decayed_confidence, initial_confidence)
        self.assertGreater(decayed_confidence, 0)


class TestIntentHistory(unittest.TestCase):
    """Test intent history and pattern detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        from intent.intent_router import IntentRouter
        self.router = IntentRouter(history_size=10)
    
    async def test_intent_history_tracking(self):
        """Test that intent history is tracked"""
        descriptions = [
            "User is coding in VS Code",
            "User is searching Stack Overflow",
            "User is back to coding",
            "User is running tests"
        ]
        
        for description in descriptions:
            intent = await self.router.classify_intent(description)
            await self.router.record_intent(intent)
        
        history = self.router.get_intent_history()
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].primary_intent, 'coding')
    
    async def test_intent_pattern_detection(self):
        """Test detection of intent patterns"""
        # Simulate coding-debugging cycle
        pattern = [
            "User is writing code",
            "User is running tests",
            "User is debugging error",
            "User is fixing code",
            "User is running tests again"
        ]
        
        for description in pattern:
            intent = await self.router.classify_intent(description)
            await self.router.record_intent(intent)
        
        detected_pattern = await self.router.detect_pattern()
        
        self.assertIsNotNone(detected_pattern)
        self.assertEqual(detected_pattern.type, 'development_cycle')
        self.assertIn('coding', detected_pattern.intents)
        self.assertIn('debugging', detected_pattern.intents)
    
    async def test_intent_transition_analysis(self):
        """Test analysis of intent transitions"""
        transitions = [
            ("User is coding", "User is searching documentation"),
            ("User is in Slack", "User is in meeting"),
            ("User is browsing", "User is coding")
        ]
        
        for from_desc, to_desc in transitions:
            from_intent = await self.router.classify_intent(from_desc)
            to_intent = await self.router.classify_intent(to_desc)
            
            transition = self.router.analyze_transition(from_intent, to_intent)
            
            self.assertIsNotNone(transition)
            self.assertIn('type', transition)
            self.assertIn('likelihood', transition)
    
    async def test_intent_summary_generation(self):
        """Test generation of intent summary"""
        # Simulate a work session
        session = [
            "User is coding" for _ in range(5)
        ] + [
            "User is in meeting" for _ in range(2)
        ] + [
            "User is browsing docs" for _ in range(3)
        ]
        
        for description in session:
            intent = await self.router.classify_intent(description)
            await self.router.record_intent(intent)
        
        summary = await self.router.generate_summary()
        
        self.assertIn('dominant_intent', summary)
        self.assertEqual(summary['dominant_intent'], 'coding')
        self.assertIn('intent_distribution', summary)
        self.assertEqual(summary['total_intents'], 10)


class TestMemoryIntegration(unittest.TestCase):
    """Test integration with memory layer"""
    
    @patch('intent.intent_router.MemoryLayer')
    async def test_intent_with_memory_context(self, mock_memory):
        """Test intent classification with memory context"""
        from intent.intent_router import IntentRouter
        
        # Mock memory returning past contexts
        mock_memory_instance = AsyncMock()
        mock_memory.return_value = mock_memory_instance
        mock_memory_instance.search_similar.return_value = [
            {'description': 'User was coding in Python', 'activity': 'coding'},
            {'description': 'User was debugging errors', 'activity': 'debugging'}
        ]
        
        router = IntentRouter(memory_layer=mock_memory_instance)
        
        # Current vague description
        current = "User is in VS Code"
        
        # Should enhance with memory
        intent = await router.classify_with_memory(current)
        
        self.assertEqual(intent.primary_intent, 'coding')
        self.assertGreater(intent.confidence, 0.7)
        self.assertTrue(intent.memory_enhanced)
    
    async def test_intent_persistence(self):
        """Test that intents are stored in memory"""
        from intent.intent_router import IntentRouter
        from memory.memory_layer import MemoryLayer
        
        memory = Mock(spec=MemoryLayer)
        memory.store_context = AsyncMock(return_value='intent_123')
        
        router = IntentRouter(memory_layer=memory)
        
        intent = await router.classify_intent("User is coding")
        await router.persist_intent(intent)
        
        memory.store_context.assert_called_once()
        stored_context = memory.store_context.call_args[0][0]
        self.assertEqual(stored_context['intent'], 'coding')
        self.assertIn('confidence', stored_context)


class TestIntentActions(unittest.TestCase):
    """Test intent-based actions and suggestions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from intent.intent_router import IntentRouter, ActionSuggester
        self.router = IntentRouter()
        self.suggester = ActionSuggester()
    
    async def test_coding_intent_suggestions(self):
        """Test suggestions for coding intent"""
        intent = await self.router.classify_intent("User is coding in Python")
        suggestions = await self.suggester.suggest_actions(intent)
        
        self.assertGreater(len(suggestions), 0)
        self.assertIn('enable_focus_mode', [s.action for s in suggestions])
        self.assertIn('mute_notifications', [s.action for s in suggestions])
    
    async def test_meeting_intent_suggestions(self):
        """Test suggestions for meeting intent"""
        intent = await self.router.classify_intent("User is in Zoom meeting")
        suggestions = await self.suggester.suggest_actions(intent)
        
        self.assertIn('close_distracting_tabs', [s.action for s in suggestions])
        self.assertIn('enable_do_not_disturb', [s.action for s in suggestions])
    
    async def test_break_intent_suggestions(self):
        """Test suggestions for break intent"""
        intent = await self.router.classify_intent("User is watching YouTube")
        context = {'duration_minutes': 30}
        
        suggestions = await self.suggester.suggest_actions(intent, context)
        
        # After 30 minutes, should suggest getting back to work
        self.assertIn('reminder_return_to_work', [s.action for s in suggestions])
    
    async def test_context_aware_suggestions(self):
        """Test that suggestions consider context"""
        intent = await self.router.classify_intent("User is browsing news")
        
        # During work hours
        work_context = {'hour': 10, 'day': 'Monday'}
        work_suggestions = await self.suggester.suggest_actions(intent, work_context)
        
        # During evening
        evening_context = {'hour': 20, 'day': 'Monday'}
        evening_suggestions = await self.suggester.suggest_actions(intent, evening_context)
        
        # Should have different suggestions
        self.assertNotEqual(
            [s.action for s in work_suggestions],
            [s.action for s in evening_suggestions]
        )


def run_tests():
    """Run all intent router tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestIntentClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestPolicyBasedRouting))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestIntentHistory))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestIntentActions))
    
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
    
    # Collect all async test methods
    test_cases = [
        TestIntentClassification(),
        TestPolicyBasedRouting(),
        TestConfidenceScoring(),
        TestIntentHistory(),
        TestMemoryIntegration(),
        TestIntentActions()
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
                    success = False
    
    sys.exit(0 if success else 1)