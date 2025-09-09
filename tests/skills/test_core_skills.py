#!/usr/bin/env python3
"""
Test Core Safe Skills using TDD.
Tests for Timer, Focus, Logger, and Status skills.
"""

import unittest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import json

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app'))


class TestTimerSkill(unittest.TestCase):
    """Test Timer/Reminder skill"""
    
    def setUp(self):
        """Set up test fixtures"""
        from skills.implementations.timer_skill import TimerSkill
        self.skill = TimerSkill()
    
    def test_timer_skill_properties(self):
        """Test timer skill basic properties"""
        from skills.core.skill_base import PermissionLevel
        
        self.assertEqual(self.skill.name, "TimerSkill")
        self.assertEqual(self.skill.permission_level, PermissionLevel.LOW)
        self.assertTrue(self.skill.reversible)
        self.assertIn("timer", self.skill.metadata.tags)
    
    async def test_create_timer(self):
        """Test creating a timer"""
        from skills.core.skill_base import SkillResult
        
        params = {
            "duration_seconds": 5,
            "message": "Test timer",
            "timer_id": "test_timer_1"
        }
        
        result = await self.skill.execute(params)
        
        self.assertTrue(result.success)
        self.assertIn("timer_id", result.data)
        self.assertEqual(result.data["timer_id"], "test_timer_1")
        self.assertIn("test_timer_1", self.skill.active_timers)
    
    async def test_cancel_timer(self):
        """Test canceling an active timer"""
        # First create a timer
        create_params = {
            "duration_seconds": 60,
            "message": "Long timer",
            "timer_id": "cancel_test"
        }
        
        create_result = await self.skill.execute(create_params)
        self.assertTrue(create_result.success)
        
        # Now cancel it
        cancel_params = {
            "action": "cancel",
            "timer_id": "cancel_test"
        }
        
        cancel_result = await self.skill.execute(cancel_params)
        
        self.assertTrue(cancel_result.success)
        self.assertNotIn("cancel_test", self.skill.active_timers)
    
    async def test_timer_completion(self):
        """Test timer completion and notification"""
        params = {
            "duration_seconds": 0.1,  # 100ms timer
            "message": "Quick timer",
            "timer_id": "quick_timer"
        }
        
        result = await self.skill.execute(params)
        self.assertTrue(result.success)
        
        # Wait for timer to complete
        await asyncio.sleep(0.2)
        
        # Timer should be removed from active timers
        self.assertNotIn("quick_timer", self.skill.active_timers)
        
        # Check that notification was triggered
        self.assertTrue(self.skill.notification_triggered)
    
    async def test_list_active_timers(self):
        """Test listing all active timers"""
        # Create multiple timers
        await self.skill.execute({"duration_seconds": 60, "message": "Timer 1", "timer_id": "t1"})
        await self.skill.execute({"duration_seconds": 60, "message": "Timer 2", "timer_id": "t2"})
        
        # List timers
        list_params = {"action": "list"}
        result = await self.skill.execute(list_params)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["timers"]), 2)
        self.assertIn("t1", [t["timer_id"] for t in result.data["timers"]])
        self.assertIn("t2", [t["timer_id"] for t in result.data["timers"]])
    
    async def test_timer_rollback(self):
        """Test rolling back timer creation"""
        params = {
            "duration_seconds": 60,
            "message": "Rollback test",
            "timer_id": "rollback_timer"
        }
        
        result = await self.skill.execute(params)
        self.assertTrue(result.success)
        self.assertIn("rollback_timer", self.skill.active_timers)
        
        # Rollback should cancel the timer
        rollback_success = await self.skill.rollback()
        self.assertTrue(rollback_success)
        self.assertNotIn("rollback_timer", self.skill.active_timers)
    
    def test_timer_validation(self):
        """Test timer parameter validation"""
        # Test with valid context
        valid_context = {
            "intent": "set_timer",
            "confidence": 0.9
        }
        valid = asyncio.run(self.skill.validate(valid_context))
        self.assertTrue(valid)
        
        # Test with low confidence
        low_confidence_context = {
            "intent": "set_timer",
            "confidence": 0.3
        }
        valid = asyncio.run(self.skill.validate(low_confidence_context))
        self.assertFalse(valid)


class TestFocusSkill(unittest.TestCase):
    """Test Focus Mode skill"""
    
    def setUp(self):
        """Set up test fixtures"""
        from skills.implementations.focus_skill import FocusSkill
        self.skill = FocusSkill()
    
    def test_focus_skill_properties(self):
        """Test focus skill basic properties"""
        from skills.core.skill_base import PermissionLevel
        
        self.assertEqual(self.skill.name, "FocusSkill")
        self.assertEqual(self.skill.permission_level, PermissionLevel.LOW)
        self.assertIn("focus", self.skill.metadata.tags)
        self.assertIn("productivity", self.skill.metadata.tags)
    
    async def test_start_focus_session(self):
        """Test starting a focus session"""
        params = {
            "action": "start",
            "duration_minutes": 25,  # Pomodoro session
            "break_reminder": True
        }
        
        result = await self.skill.execute(params)
        
        self.assertTrue(result.success)
        self.assertTrue(self.skill.session_active)
        self.assertIsNotNone(self.skill.current_session)
        self.assertEqual(self.skill.current_session["duration_minutes"], 25)
    
    async def test_end_focus_session(self):
        """Test ending a focus session"""
        # Start a session first
        start_params = {
            "action": "start",
            "duration_minutes": 25
        }
        await self.skill.execute(start_params)
        
        # End the session
        end_params = {"action": "end"}
        result = await self.skill.execute(end_params)
        
        self.assertTrue(result.success)
        self.assertFalse(self.skill.session_active)
        self.assertIsNone(self.skill.current_session)
        self.assertIn("duration", result.data)
    
    async def test_focus_break_reminder(self):
        """Test break reminder during focus session"""
        params = {
            "action": "start",
            "duration_minutes": 0.05,  # 3 seconds for testing
            "break_reminder": True
        }
        
        result = await self.skill.execute(params)
        self.assertTrue(result.success)
        
        # Wait for break reminder
        await asyncio.sleep(0.1)
        
        # Check that break reminder was triggered
        self.assertTrue(self.skill.break_reminder_triggered)
    
    async def test_focus_statistics(self):
        """Test getting focus session statistics"""
        # Complete a few sessions
        for i in range(3):
            await self.skill.execute({"action": "start", "duration_minutes": 25})
            await asyncio.sleep(0.01)  # Brief pause
            await self.skill.execute({"action": "end"})
        
        # Get statistics
        stats_params = {"action": "stats"}
        result = await self.skill.execute(stats_params)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["total_sessions"], 3)
        self.assertIn("total_focus_time", result.data)
        self.assertIn("average_session_length", result.data)


class TestLoggerSkill(unittest.TestCase):
    """Test Activity Logger skill"""
    
    def setUp(self):
        """Set up test fixtures"""
        from skills.implementations.logger_skill import LoggerSkill
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.skill = LoggerSkill(persist_directory=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logger_skill_properties(self):
        """Test logger skill basic properties"""
        from skills.core.skill_base import PermissionLevel
        
        self.assertEqual(self.skill.name, "LoggerSkill")
        self.assertEqual(self.skill.permission_level, PermissionLevel.LOW)
        self.assertIn("logging", self.skill.metadata.tags)
    
    async def test_log_activity(self):
        """Test logging an activity"""
        params = {
            "activity": "coding",
            "description": "Working on Porter.AI",
            "application": "VS Code",
            "tags": ["python", "ai"]
        }
        
        result = await self.skill.execute(params)
        
        self.assertTrue(result.success)
        self.assertIn("activity_id", result.data)
        self.assertIsNotNone(result.data["activity_id"])
    
    async def test_query_activities(self):
        """Test querying logged activities"""
        # Log some activities
        activities = [
            {"activity": "coding", "description": "Python development"},
            {"activity": "meeting", "description": "Team standup"},
            {"activity": "research", "description": "Reading papers"}
        ]
        
        for activity in activities:
            await self.skill.execute(activity)
        
        # Query activities
        query_params = {
            "action": "query",
            "query": "coding"
        }
        
        result = await self.skill.execute(query_params)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.data["activities"]), 0)
        self.assertIn("coding", result.data["activities"][0]["activity"])
    
    async def test_activity_summary(self):
        """Test generating activity summary"""
        # Log activities
        for i in range(5):
            await self.skill.execute({
                "activity": "coding" if i % 2 == 0 else "meeting",
                "description": f"Activity {i}"
            })
        
        # Get summary
        summary_params = {"action": "summary", "period": "day"}
        result = await self.skill.execute(summary_params)
        
        self.assertTrue(result.success)
        self.assertIn("total_activities", result.data)
        self.assertEqual(result.data["total_activities"], 5)
        self.assertIn("activity_breakdown", result.data)
        self.assertIn("coding", result.data["activity_breakdown"])
    
    async def test_clear_activities(self):
        """Test clearing activity history"""
        # Log some activities
        await self.skill.execute({"activity": "test", "description": "Test activity"})
        
        # Clear activities
        clear_params = {"action": "clear", "confirm": True}
        result = await self.skill.execute(clear_params)
        
        self.assertTrue(result.success)
        
        # Verify activities are cleared
        query_result = await self.skill.execute({"action": "query", "query": "test"})
        self.assertEqual(len(query_result.data["activities"]), 0)


class TestStatusSkill(unittest.TestCase):
    """Test System Status skill"""
    
    def setUp(self):
        """Set up test fixtures"""
        from skills.implementations.status_skill import StatusSkill
        self.skill = StatusSkill()
    
    def test_status_skill_properties(self):
        """Test status skill basic properties"""
        from skills.core.skill_base import PermissionLevel
        
        self.assertEqual(self.skill.name, "StatusSkill")
        self.assertEqual(self.skill.permission_level, PermissionLevel.LOW)
        self.assertIn("monitoring", self.skill.metadata.tags)
    
    async def test_get_system_status(self):
        """Test getting system status"""
        params = {"type": "system"}
        
        result = await self.skill.execute(params)
        
        self.assertTrue(result.success)
        self.assertIn("cpu_percent", result.data)
        self.assertIn("memory_percent", result.data)
        self.assertIn("disk_usage", result.data)
        self.assertIsInstance(result.data["cpu_percent"], (int, float))
    
    async def test_get_porter_status(self):
        """Test getting Porter.AI component status"""
        params = {"type": "porter"}
        
        result = await self.skill.execute(params)
        
        self.assertTrue(result.success)
        self.assertIn("components", result.data)
        self.assertIn("perception", result.data["components"])
        self.assertIn("memory", result.data["components"])
        self.assertIn("skills", result.data["components"])
    
    async def test_get_active_skills(self):
        """Test getting list of active skills"""
        params = {"type": "skills"}
        
        result = await self.skill.execute(params)
        
        self.assertTrue(result.success)
        self.assertIn("active_skills", result.data)
        self.assertIn("registered_skills", result.data)
        self.assertIsInstance(result.data["active_skills"], list)
    
    async def test_get_performance_metrics(self):
        """Test getting performance metrics"""
        params = {"type": "performance"}
        
        result = await self.skill.execute(params)
        
        self.assertTrue(result.success)
        self.assertIn("response_time_ms", result.data)
        self.assertIn("throughput", result.data)
        self.assertIn("error_rate", result.data)


def run_async_tests():
    """Run all async tests"""
    test_cases = [
        TestTimerSkill(),
        TestFocusSkill(),
        TestLoggerSkill(),
        TestStatusSkill()
    ]
    
    success = True
    
    for test_case in test_cases:
        print(f"\n--- Running {test_case.__class__.__name__} ---")
        
        # Set up test case
        if hasattr(test_case, 'setUp'):
            test_case.setUp()
        
        # Run each async test method
        test_methods = [method for method in dir(test_case) 
                       if method.startswith('test_') and 
                       asyncio.iscoroutinefunction(getattr(test_case, method))]
        
        for method_name in test_methods:
            try:
                print(f"Running {method_name}...")
                test_method = getattr(test_case, method_name)
                asyncio.run(test_method())
                print(f"✓ {method_name} passed")
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                success = False
        
        # Tear down test case
        if hasattr(test_case, 'tearDown'):
            test_case.tearDown()
    
    return success


if __name__ == '__main__':
    # Run standard unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTimerSkill))
    suite.addTests(loader.loadTestsFromTestCase(TestFocusSkill))
    suite.addTests(loader.loadTestsFromTestCase(TestLoggerSkill))
    suite.addTests(loader.loadTestsFromTestCase(TestStatusSkill))
    
    # Run standard tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run async tests
    print("\n" + "="*50)
    print("Running Async Tests")
    print("="*50)
    
    async_success = run_async_tests()
    
    # Exit with appropriate code
    sys.exit(0 if (result.wasSuccessful() and async_success) else 1)