#!/usr/bin/env python3
"""
Skill Executor for safe skill execution with rate limiting and auditing.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .skill_base import Skill, SkillResult
from .permission_manager import PermissionManager

logger = logging.getLogger(__name__)


class SkillExecutor:
    """
    Executes skills safely with rate limiting, timeouts, and auditing.
    """
    
    def __init__(
        self,
        permission_manager: Optional[PermissionManager] = None,
        max_executions_per_minute: int = 30,
        rate_limit_window: int = 60,
        timeout_seconds: int = 30
    ):
        """
        Initialize skill executor.
        
        Args:
            permission_manager: Permission manager instance
            max_executions_per_minute: Max executions per minute
            rate_limit_window: Rate limit window in seconds
            timeout_seconds: Default timeout for skill execution
        """
        self.permission_manager = permission_manager or PermissionManager()
        self.max_executions_per_minute = max_executions_per_minute
        self.rate_limit_window = rate_limit_window
        self.timeout_seconds = timeout_seconds
        
        # Rate limiting
        self._execution_times: Dict[str, deque] = defaultdict(deque)
        
        # Audit log
        self._audit_log: List[Dict] = []
        self._max_audit_entries = 1000
    
    async def execute_safe(
        self,
        skill: Skill,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SkillResult:
        """
        Execute a skill safely with all protections.
        
        Args:
            skill: Skill to execute
            params: Execution parameters
            context: Optional execution context
            
        Returns:
            SkillResult with outcome
        """
        start_time = time.time()
        audit_entry = {
            'skill_name': skill.name,
            'params': params.copy(),
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        
        try:
            # Check rate limit
            if not self._check_rate_limit(skill.name):
                error_msg = f"Rate limit exceeded for {skill.name}"
                logger.warning(error_msg)
                result = SkillResult(success=False, error=error_msg)
                audit_entry['success'] = False
                audit_entry['error'] = error_msg
                return result
            
            # Check permissions
            if not await self.permission_manager.request_permission(skill):
                error_msg = f"Permission denied for {skill.name}"
                logger.warning(error_msg)
                result = SkillResult(success=False, error=error_msg)
                audit_entry['success'] = False
                audit_entry['error'] = error_msg
                return result
            
            # Validate context
            if context and not await skill.validate(context):
                error_msg = f"Validation failed for {skill.name}"
                logger.warning(error_msg)
                result = SkillResult(success=False, error=error_msg)
                audit_entry['success'] = False
                audit_entry['error'] = error_msg
                return result
            
            # Execute with timeout
            timeout = skill.max_execution_time if hasattr(skill, 'max_execution_time') else self.timeout_seconds
            result = await self.execute_with_timeout(skill, params, timeout)
            
            # Handle failure with rollback
            if not result.success and skill.reversible:
                logger.info(f"Attempting rollback for {skill.name}")
                try:
                    rollback_success = await asyncio.wait_for(
                        skill.rollback(),
                        timeout=timeout
                    )
                    if rollback_success:
                        logger.info(f"Rollback successful for {skill.name}")
                        result.message = (result.message or "") + " (rolled back)"
                    else:
                        logger.error(f"Rollback failed for {skill.name}")
                except Exception as e:
                    logger.error(f"Rollback error for {skill.name}: {e}")
            
            # Update audit
            audit_entry['success'] = result.success
            audit_entry['result_data'] = result.data
            audit_entry['error'] = result.error
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error executing {skill.name}: {e}"
            logger.error(error_msg)
            
            # Try rollback on error
            if skill.reversible:
                try:
                    await skill.rollback()
                    logger.info(f"Emergency rollback for {skill.name}")
                except:
                    pass
            
            audit_entry['success'] = False
            audit_entry['error'] = str(e)
            
            return SkillResult(success=False, error=error_msg)
            
        finally:
            # Record execution time
            audit_entry['duration_ms'] = int((time.time() - start_time) * 1000)
            self._add_audit_entry(audit_entry)
            
            # Record for rate limiting
            self._record_execution(skill.name)
    
    async def execute_with_timeout(
        self,
        skill: Skill,
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> SkillResult:
        """
        Execute skill with timeout.
        
        Args:
            skill: Skill to execute
            params: Execution parameters
            timeout: Timeout in seconds
            
        Returns:
            SkillResult
        """
        timeout = timeout or self.timeout_seconds
        
        try:
            # Call pre-execute hook
            if not await asyncio.wait_for(skill.pre_execute(params), timeout=5):
                return SkillResult(success=False, error="Pre-execution check failed")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                skill.execute(params),
                timeout=timeout
            )
            
            # Call post-execute hook
            await asyncio.wait_for(skill.post_execute(result), timeout=5)
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Skill {skill.name} timed out after {timeout} seconds"
            logger.error(error_msg)
            return SkillResult(success=False, error=error_msg)
        
        except Exception as e:
            error_msg = f"Error executing {skill.name}: {e}"
            logger.error(error_msg)
            return SkillResult(success=False, error=str(e))
    
    def _check_rate_limit(self, skill_name: str) -> bool:
        """
        Check if skill execution is within rate limit.
        
        Args:
            skill_name: Name of skill
            
        Returns:
            True if within limit
        """
        now = time.time()
        
        # Clean old entries
        execution_times = self._execution_times[skill_name]
        while execution_times and execution_times[0] < now - self.rate_limit_window:
            execution_times.popleft()
        
        # Check limit
        return len(execution_times) < self.max_executions_per_minute
    
    def _record_execution(self, skill_name: str) -> None:
        """
        Record skill execution for rate limiting.
        
        Args:
            skill_name: Name of skill
        """
        self._execution_times[skill_name].append(time.time())
    
    def _add_audit_entry(self, entry: Dict) -> None:
        """
        Add entry to audit log.
        
        Args:
            entry: Audit entry
        """
        self._audit_log.append(entry)
        
        # Trim if too large
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]
    
    def get_audit_log(
        self,
        skill_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get audit log entries.
        
        Args:
            skill_name: Filter by skill name
            limit: Maximum entries to return
            
        Returns:
            List of audit entries
        """
        if skill_name:
            entries = [e for e in self._audit_log if e['skill_name'] == skill_name]
        else:
            entries = self._audit_log
        
        return entries[-limit:]
    
    def get_execution_stats(self) -> Dict[str, Dict]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary of statistics per skill
        """
        stats = {}
        
        for entry in self._audit_log:
            skill_name = entry['skill_name']
            
            if skill_name not in stats:
                stats[skill_name] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'avg_duration_ms': 0,
                    'total_duration_ms': 0
                }
            
            stats[skill_name]['total'] += 1
            
            if entry.get('success'):
                stats[skill_name]['successful'] += 1
            else:
                stats[skill_name]['failed'] += 1
            
            duration = entry.get('duration_ms', 0)
            stats[skill_name]['total_duration_ms'] += duration
        
        # Calculate averages
        for skill_stats in stats.values():
            if skill_stats['total'] > 0:
                skill_stats['avg_duration_ms'] = (
                    skill_stats['total_duration_ms'] / skill_stats['total']
                )
        
        return stats
    
    def clear_audit_log(self) -> None:
        """Clear audit log"""
        self._audit_log.clear()
        logger.info("Cleared audit log")
    
    def reset_rate_limits(self) -> None:
        """Reset all rate limits"""
        self._execution_times.clear()
        logger.info("Reset rate limits")
    
    def __repr__(self) -> str:
        """Developer representation"""
        return (f"<SkillExecutor: {len(self._audit_log)} audit entries, "
                f"rate_limit={self.max_executions_per_minute}/min>")