#!/usr/bin/env python3
"""
Timer/Reminder Skill for Porter.AI.
Provides timer and reminder functionality with notifications.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from ..core.skill_base import Skill, SkillResult, PermissionLevel, SkillMetadata, SkillRequirements

logger = logging.getLogger(__name__)


class Timer:
    """Represents an active timer"""
    
    def __init__(self, timer_id: str, duration_seconds: float, message: str):
        self.timer_id = timer_id
        self.duration_seconds = duration_seconds
        self.message = message
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(seconds=duration_seconds)
        self.task: Optional[asyncio.Task] = None
        self.completed = False
        self.cancelled = False
    
    def time_remaining(self) -> float:
        """Get time remaining in seconds"""
        if self.completed or self.cancelled:
            return 0
        remaining = (self.end_time - datetime.now()).total_seconds()
        return max(0, remaining)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert timer to dictionary"""
        return {
            "timer_id": self.timer_id,
            "message": self.message,
            "duration_seconds": self.duration_seconds,
            "time_remaining": self.time_remaining(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "completed": self.completed,
            "cancelled": self.cancelled
        }


class TimerSkill(Skill):
    """
    Timer/Reminder skill for setting timers and reminders.
    Supports multiple concurrent timers with notifications.
    """
    
    def __init__(self):
        """Initialize timer skill"""
        super().__init__()
        self.name = "TimerSkill"
        self.description = "Set timers and reminders"
        self.permission_level = PermissionLevel.LOW
        self.reversible = True
        
        # Metadata
        self.metadata = SkillMetadata(
            name=self.name,
            description=self.description,
            category="productivity",
            tags=["timer", "reminder", "notification", "time-management"],
            version="1.0.0",
            author="Porter.AI"
        )
        
        # Requirements
        self.requirements = SkillRequirements(
            min_confidence=0.7,
            required_intents=["set_timer", "set_reminder", "cancel_timer"],
            required_context_keys=[],
            cooldown_seconds=0
        )
        
        # Active timers
        self.active_timers: Dict[str, Timer] = {}
        self.notification_triggered = False  # For testing
        self._last_timer_id: Optional[str] = None
    
    async def validate(self, context: Dict[str, Any]) -> bool:
        """
        Validate timer skill can be executed.
        
        Args:
            context: Execution context
            
        Returns:
            True if valid
        """
        # Check confidence
        if context.get("confidence", 0) < self.requirements.min_confidence:
            logger.debug(f"Confidence too low: {context.get('confidence', 0)}")
            return False
        
        # Check intent
        intent = context.get("intent", "")
        if intent and intent not in self.requirements.required_intents:
            # Allow if no specific intent required
            if self.requirements.required_intents:
                return False
        
        return True
    
    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        """
        Execute timer action.
        
        Args:
            params: Execution parameters
                - action: "create" (default), "cancel", "list"
                - duration_seconds: Timer duration (for create)
                - message: Timer message (for create)
                - timer_id: Optional timer ID (for create/cancel)
                
        Returns:
            SkillResult with timer information
        """
        action = params.get("action", "create")
        
        try:
            if action == "create" or "duration_seconds" in params:
                return await self._create_timer(params)
            elif action == "cancel":
                return await self._cancel_timer(params)
            elif action == "list":
                return await self._list_timers()
            else:
                return SkillResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"Timer execution error: {e}")
            return SkillResult(
                success=False,
                error=str(e)
            )
    
    async def _create_timer(self, params: Dict[str, Any]) -> SkillResult:
        """Create a new timer"""
        duration = params.get("duration_seconds", 300)  # Default 5 minutes
        message = params.get("message", "Timer completed!")
        timer_id = params.get("timer_id", str(uuid.uuid4()))
        
        # Create timer object
        timer = Timer(timer_id, duration, message)
        
        # Start timer task
        timer.task = asyncio.create_task(self._run_timer(timer))
        
        # Store timer
        self.active_timers[timer_id] = timer
        self._last_timer_id = timer_id
        
        logger.info(f"Created timer {timer_id}: {message} ({duration}s)")
        
        return SkillResult(
            success=True,
            data={
                "timer_id": timer_id,
                "message": message,
                "duration_seconds": duration,
                "end_time": timer.end_time.isoformat()
            },
            message=f"Timer set for {duration} seconds",
            rollback_data=timer_id
        )
    
    async def _cancel_timer(self, params: Dict[str, Any]) -> SkillResult:
        """Cancel an active timer"""
        timer_id = params.get("timer_id")
        
        if not timer_id:
            return SkillResult(
                success=False,
                error="No timer_id provided"
            )
        
        if timer_id not in self.active_timers:
            return SkillResult(
                success=False,
                error=f"Timer {timer_id} not found"
            )
        
        timer = self.active_timers[timer_id]
        
        # Cancel the timer task
        if timer.task and not timer.task.done():
            timer.task.cancel()
        
        timer.cancelled = True
        
        # Remove from active timers
        del self.active_timers[timer_id]
        
        logger.info(f"Cancelled timer {timer_id}")
        
        return SkillResult(
            success=True,
            message=f"Timer {timer_id} cancelled"
        )
    
    async def _list_timers(self) -> SkillResult:
        """List all active timers"""
        timers = [
            timer.to_dict() 
            for timer in self.active_timers.values()
            if not timer.completed and not timer.cancelled
        ]
        
        return SkillResult(
            success=True,
            data={"timers": timers},
            message=f"Found {len(timers)} active timer(s)"
        )
    
    async def _run_timer(self, timer: Timer) -> None:
        """Run a timer to completion"""
        try:
            # Wait for timer duration
            await asyncio.sleep(timer.duration_seconds)
            
            # Mark as completed
            timer.completed = True
            
            # Trigger notification
            await self._trigger_notification(timer)
            
            # Remove from active timers
            if timer.timer_id in self.active_timers:
                del self.active_timers[timer.timer_id]
            
            logger.info(f"Timer {timer.timer_id} completed: {timer.message}")
            
        except asyncio.CancelledError:
            logger.debug(f"Timer {timer.timer_id} was cancelled")
        except Exception as e:
            logger.error(f"Timer {timer.timer_id} error: {e}")
    
    async def _trigger_notification(self, timer: Timer) -> None:
        """
        Trigger notification for completed timer.
        In a real implementation, this would show a system notification.
        """
        self.notification_triggered = True
        
        # Log notification (in real app, would show system notification)
        logger.info(f"🔔 NOTIFICATION: {timer.message}")
        
        # In a real implementation, you might:
        # - Show a system notification (using plyer or similar)
        # - Play a sound
        # - Send to UI via WebSocket
        # - Log to activity history
    
    async def rollback(self) -> bool:
        """
        Rollback last timer creation.
        
        Returns:
            True if successful
        """
        if not self._last_timer_id:
            return True
        
        # Cancel the last created timer
        if self._last_timer_id in self.active_timers:
            result = await self._cancel_timer({"timer_id": self._last_timer_id})
            self._last_timer_id = None
            return result.success
        
        return True
    
    def cleanup(self) -> None:
        """Clean up all active timers"""
        for timer in self.active_timers.values():
            if timer.task and not timer.task.done():
                timer.task.cancel()
        
        self.active_timers.clear()
        logger.info("Cleaned up all timers")