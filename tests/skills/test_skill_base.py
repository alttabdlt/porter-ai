#!/usr/bin/env python3
"""
Test Skills Layer foundation using TDD.
Tests for base Skill class, permissions, and execution safety.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path
from datetime import datetime
import time

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app'))


class TestSkillBase(unittest.TestCase):
    """Test base Skill class and lifecycle"""
    
    def test_skill_abstract_class(self):
        """Test that Skill is abstract and cannot be instantiated"""
        from skills.core.skill_base import Skill
        
        # Should not be able to instantiate abstract class
        with self.assertRaises(TypeError):
            skill = Skill()
    
    def test_skill_implementation(self):
        """Test implementing a concrete skill"""
        from skills.core.skill_base import Skill, PermissionLevel
        
        class TestSkill(Skill):
            def __init__(self):
                super().__init__()
                self.name = "TestSkill"
                self.description = "A test skill"
                self.permission_level = PermissionLevel.LOW
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                from skills.core.skill_base import SkillResult
                return SkillResult(success=True, data={"test": "result"})
                
            async def rollback(self):
                return True
        
        skill = TestSkill()
        self.assertEqual(skill.name, "TestSkill")
        self.assertEqual(skill.permission_level.value, "low")
        self.assertTrue(skill.reversible)
    
    async def test_validate_execute_rollback_lifecycle(self):
        """Test the full skill lifecycle"""
        from skills.core.skill_base import Skill, SkillResult, PermissionLevel
        
        class LifecycleSkill(Skill):
            def __init__(self):
                super().__init__()
                self.name = "LifecycleSkill"
                self.validated = False
                self.executed = False
                self.rolled_back = False
                
            async def validate(self, context):
                self.validated = True
                return context.get('valid', True)
                
            async def execute(self, params):
                self.executed = True
                return SkillResult(
                    success=params.get('success', True),
                    data={"executed": True}
                )
                
            async def rollback(self):
                self.rolled_back = True
                return True
        
        skill = LifecycleSkill()
        
        # Test validation
        valid = await skill.validate({'valid': True})
        self.assertTrue(valid)
        self.assertTrue(skill.validated)
        
        # Test execution
        result = await skill.execute({'success': True})
        self.assertTrue(result.success)
        self.assertTrue(skill.executed)
        
        # Test rollback
        rolled_back = await skill.rollback()
        self.assertTrue(rolled_back)
        self.assertTrue(skill.rolled_back)
    
    def test_skill_result_dataclass(self):
        """Test SkillResult dataclass"""
        from skills.core.skill_base import SkillResult
        
        # Success result
        result = SkillResult(
            success=True,
            data={"key": "value"},
            message="Operation completed"
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.data["key"], "value")
        self.assertEqual(result.message, "Operation completed")
        self.assertIsNotNone(result.timestamp)
        
        # Failure result
        error_result = SkillResult(
            success=False,
            error="Something went wrong",
            message="Operation failed"
        )
        
        self.assertFalse(error_result.success)
        self.assertEqual(error_result.error, "Something went wrong")
    
    def test_permission_levels(self):
        """Test permission level enumeration"""
        from skills.core.skill_base import PermissionLevel
        
        self.assertEqual(PermissionLevel.LOW.value, "low")
        self.assertEqual(PermissionLevel.MEDIUM.value, "medium")
        self.assertEqual(PermissionLevel.HIGH.value, "high")
        
        # Test that they are distinct
        self.assertNotEqual(PermissionLevel.LOW, PermissionLevel.MEDIUM)
        self.assertNotEqual(PermissionLevel.MEDIUM, PermissionLevel.HIGH)
        self.assertNotEqual(PermissionLevel.LOW, PermissionLevel.HIGH)


class TestSkillMetadata(unittest.TestCase):
    """Test skill metadata and properties"""
    
    def test_skill_metadata(self):
        """Test skill metadata attributes"""
        from skills.core.skill_base import Skill, SkillMetadata, PermissionLevel
        
        metadata = SkillMetadata(
            name="TestSkill",
            description="A skill for testing",
            category="testing",
            tags=["test", "demo"],
            version="1.0.0",
            author="TestAuthor"
        )
        
        self.assertEqual(metadata.name, "TestSkill")
        self.assertEqual(metadata.category, "testing")
        self.assertIn("test", metadata.tags)
        self.assertEqual(metadata.version, "1.0.0")
    
    def test_skill_with_metadata(self):
        """Test skill with full metadata"""
        from skills.core.skill_base import Skill, SkillMetadata, PermissionLevel, SkillResult
        
        class MetadataSkill(Skill):
            def __init__(self):
                super().__init__()
                self.metadata = SkillMetadata(
                    name="MetadataSkill",
                    description="Skill with metadata",
                    category="productivity",
                    tags=["timer", "focus"],
                    version="1.0.0"
                )
                self.permission_level = PermissionLevel.LOW
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                return SkillResult(success=True)
                
            async def rollback(self):
                return True
        
        skill = MetadataSkill()
        self.assertEqual(skill.metadata.name, "MetadataSkill")
        self.assertEqual(skill.metadata.category, "productivity")
    
    def test_skill_requirements(self):
        """Test skill requirements and dependencies"""
        from skills.core.skill_base import Skill, SkillRequirements
        
        requirements = SkillRequirements(
            min_confidence=0.8,
            required_intents=["coding", "debugging"],
            required_context_keys=["application", "window_title"],
            cooldown_seconds=30
        )
        
        self.assertEqual(requirements.min_confidence, 0.8)
        self.assertIn("coding", requirements.required_intents)
        self.assertEqual(requirements.cooldown_seconds, 30)


class TestSkillSafety(unittest.TestCase):
    """Test skill safety features"""
    
    def test_reversible_skill(self):
        """Test reversible skill property"""
        from skills.core.skill_base import Skill, PermissionLevel, SkillResult
        
        class ReversibleSkill(Skill):
            def __init__(self):
                super().__init__()
                self.reversible = True
                self.state_before = None
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                self.state_before = params.get('current_state')
                return SkillResult(success=True, rollback_data=self.state_before)
                
            async def rollback(self):
                # Restore previous state
                return self.state_before is not None
        
        skill = ReversibleSkill()
        self.assertTrue(skill.reversible)
        
        # Execute and save state
        result = asyncio.run(skill.execute({'current_state': 'original'}))
        self.assertEqual(result.rollback_data, 'original')
        
        # Rollback should work
        can_rollback = asyncio.run(skill.rollback())
        self.assertTrue(can_rollback)
    
    def test_confirmation_required(self):
        """Test skills that require confirmation"""
        from skills.core.skill_base import Skill, PermissionLevel, SkillResult
        
        class DangerousSkill(Skill):
            def __init__(self):
                super().__init__()
                self.permission_level = PermissionLevel.HIGH
                self.requires_confirmation = True
                
            async def validate(self, context):
                return context.get('user_confirmed', False)
                
            async def execute(self, params):
                return SkillResult(success=True)
                
            async def rollback(self):
                return False  # Cannot rollback
        
        skill = DangerousSkill()
        self.assertTrue(skill.requires_confirmation)
        self.assertEqual(skill.permission_level, PermissionLevel.HIGH)
        
        # Should not validate without confirmation
        valid = asyncio.run(skill.validate({}))
        self.assertFalse(valid)
        
        # Should validate with confirmation
        valid = asyncio.run(skill.validate({'user_confirmed': True}))
        self.assertTrue(valid)
    
    async def test_skill_timeout(self):
        """Test skill execution timeout"""
        from skills.core.skill_base import Skill, SkillResult, PermissionLevel, SkillExecutor
        
        class SlowSkill(Skill):
            def __init__(self):
                super().__init__()
                self.max_execution_time = 1  # 1 second timeout
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                # Simulate slow operation
                await asyncio.sleep(2)
                return SkillResult(success=True)
                
            async def rollback(self):
                return True
        
        skill = SlowSkill()
        executor = SkillExecutor(timeout_seconds=1)
        
        # Should timeout
        result = await executor.execute_with_timeout(skill, {})
        self.assertFalse(result.success)
        self.assertIn("timeout", result.error.lower())


class TestSkillRegistry(unittest.TestCase):
    """Test skill registry and discovery"""
    
    def test_skill_registration(self):
        """Test registering skills in registry"""
        from skills.core.skill_registry import SkillRegistry
        from skills.core.skill_base import Skill, PermissionLevel, SkillResult
        
        registry = SkillRegistry()
        
        class TestSkill(Skill):
            def __init__(self):
                super().__init__()
                self.name = "TestSkill"
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                return SkillResult(success=True)
                
            async def rollback(self):
                return True
        
        skill = TestSkill()
        registry.register(skill)
        
        # Should be able to retrieve
        retrieved = registry.get("TestSkill")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "TestSkill")
    
    def test_skill_discovery(self):
        """Test automatic skill discovery from directory"""
        from skills.core.skill_registry import SkillRegistry
        
        registry = SkillRegistry()
        
        # Should discover skills from directory
        discovered = registry.discover_skills("skills/implementations")
        
        # Note: This will fail until we have actual skill implementations
        # For now, just test the method exists
        self.assertIsInstance(discovered, list)
    
    def test_skill_categorization(self):
        """Test getting skills by category"""
        from skills.core.skill_registry import SkillRegistry
        from skills.core.skill_base import Skill, SkillMetadata, PermissionLevel, SkillResult
        
        registry = SkillRegistry()
        
        class ProductivitySkill(Skill):
            def __init__(self):
                super().__init__()
                self.metadata = SkillMetadata(
                    name="ProductivitySkill",
                    category="productivity"
                )
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                return SkillResult(success=True)
                
            async def rollback(self):
                return True
        
        skill = ProductivitySkill()
        registry.register(skill)
        
        # Get by category
        productivity_skills = registry.get_by_category("productivity")
        self.assertEqual(len(productivity_skills), 1)
        self.assertEqual(productivity_skills[0].metadata.name, "ProductivitySkill")


class TestPermissionManager(unittest.TestCase):
    """Test permission management system"""
    
    def test_permission_check(self):
        """Test permission checking"""
        from skills.core.permission_manager import PermissionManager
        from skills.core.skill_base import PermissionLevel
        
        manager = PermissionManager()
        
        # Set user permissions
        manager.set_permission("TestSkill", PermissionLevel.MEDIUM)
        
        # Check permissions
        self.assertTrue(manager.has_permission("TestSkill", PermissionLevel.LOW))
        self.assertTrue(manager.has_permission("TestSkill", PermissionLevel.MEDIUM))
        self.assertFalse(manager.has_permission("TestSkill", PermissionLevel.HIGH))
    
    async def test_permission_request(self):
        """Test requesting permission from user"""
        from skills.core.permission_manager import PermissionManager
        from skills.core.skill_base import Skill, PermissionLevel, SkillResult
        
        manager = PermissionManager()
        
        class TestSkill(Skill):
            def __init__(self):
                super().__init__()
                self.name = "TestSkill"
                self.permission_level = PermissionLevel.MEDIUM
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                return SkillResult(success=True)
                
            async def rollback(self):
                return True
        
        skill = TestSkill()
        
        # Mock user approval
        manager.auto_approve = True  # For testing
        approved = await manager.request_permission(skill)
        self.assertTrue(approved)
        
        # Mock user denial
        manager.auto_approve = False
        approved = await manager.request_permission(skill)
        self.assertFalse(approved)
    
    def test_permission_persistence(self):
        """Test that permissions are saved and loaded"""
        from skills.core.permission_manager import PermissionManager
        from skills.core.skill_base import PermissionLevel
        import tempfile
        import json
        
        # Create temp file for permissions
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        manager1 = PermissionManager(permission_file=temp_file)
        manager1.set_permission("SkillA", PermissionLevel.HIGH)
        manager1.set_permission("SkillB", PermissionLevel.LOW)
        manager1.save()
        
        # Load in new manager
        manager2 = PermissionManager(permission_file=temp_file)
        manager2.load()
        
        self.assertTrue(manager2.has_permission("SkillA", PermissionLevel.HIGH))
        self.assertTrue(manager2.has_permission("SkillB", PermissionLevel.LOW))
        
        # Cleanup
        import os
        os.unlink(temp_file)


class TestSkillExecutor(unittest.TestCase):
    """Test safe skill execution"""
    
    async def test_rate_limiting(self):
        """Test that skills are rate limited"""
        from skills.core.skill_executor import SkillExecutor
        from skills.core.skill_base import Skill, PermissionLevel, SkillResult
        
        class FastSkill(Skill):
            def __init__(self):
                super().__init__()
                self.name = "FastSkill"
                self.execution_count = 0
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                self.execution_count += 1
                return SkillResult(success=True)
                
            async def rollback(self):
                return True
        
        executor = SkillExecutor(
            max_executions_per_minute=5,
            rate_limit_window=60
        )
        
        skill = FastSkill()
        
        # Execute multiple times quickly
        results = []
        for i in range(10):
            result = await executor.execute_safe(skill, {})
            results.append(result)
        
        # Should have rate limited after 5
        successful = sum(1 for r in results if r.success)
        self.assertEqual(successful, 5)
        
        # Later executions should be rate limited
        self.assertFalse(results[-1].success)
        self.assertIn("rate limit", results[-1].error.lower())
    
    async def test_execution_logging(self):
        """Test that all executions are logged"""
        from skills.core.skill_executor import SkillExecutor
        from skills.core.skill_base import Skill, PermissionLevel, SkillResult
        
        class LoggedSkill(Skill):
            def __init__(self):
                super().__init__()
                self.name = "LoggedSkill"
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                return SkillResult(success=True, data={"result": "test"})
                
            async def rollback(self):
                return True
        
        executor = SkillExecutor()
        skill = LoggedSkill()
        
        # Execute skill
        result = await executor.execute_safe(skill, {"param": "value"})
        
        # Check audit log
        log_entries = executor.get_audit_log()
        self.assertEqual(len(log_entries), 1)
        
        entry = log_entries[0]
        self.assertEqual(entry['skill_name'], "LoggedSkill")
        self.assertEqual(entry['params']['param'], "value")
        self.assertTrue(entry['success'])
        self.assertIn('timestamp', entry)
        self.assertIn('duration_ms', entry)
    
    async def test_rollback_on_error(self):
        """Test automatic rollback on execution error"""
        from skills.core.skill_executor import SkillExecutor
        from skills.core.skill_base import Skill, PermissionLevel, SkillResult
        
        class FailingSkill(Skill):
            def __init__(self):
                super().__init__()
                self.name = "FailingSkill"
                self.reversible = True
                self.changes_made = False
                self.rolled_back = False
                
            async def validate(self, context):
                return True
                
            async def execute(self, params):
                self.changes_made = True
                if params.get('fail', False):
                    raise Exception("Execution failed")
                return SkillResult(success=True)
                
            async def rollback(self):
                self.rolled_back = True
                self.changes_made = False
                return True
        
        executor = SkillExecutor()
        skill = FailingSkill()
        
        # Execute with failure
        result = await executor.execute_safe(skill, {"fail": True})
        
        # Should have rolled back
        self.assertFalse(result.success)
        self.assertTrue(skill.rolled_back)
        self.assertFalse(skill.changes_made)


def run_tests():
    """Run all skill base tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSkillBase))
    suite.addTests(loader.loadTestsFromTestCase(TestSkillMetadata))
    suite.addTests(loader.loadTestsFromTestCase(TestSkillSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestSkillRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestPermissionManager))
    suite.addTests(loader.loadTestsFromTestCase(TestSkillExecutor))
    
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
        TestSkillBase(),
        TestSkillSafety(),
        TestSkillExecutor()
    ]
    
    print("\n--- Running Async Tests ---")
    
    for test_case in test_cases:
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