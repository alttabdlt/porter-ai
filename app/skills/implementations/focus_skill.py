#!/usr/bin/env python3
"""
Focus Mode Skill for Porter.AI.
Manages focus sessions, break reminders, and productivity tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from ..core.skill_base import Skill, SkillResult, PermissionLevel, SkillMetadata, SkillRequirements

logger = logging.getLogger(__name__)


class FocusSession:
    """Represents a focus session"""
    
    def __init__(self, duration_minutes: int, break_reminder: bool = True):
        self.duration_minutes = duration_minutes
        self.break_reminder = break_reminder
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)
        self.actual_end_time: Optional[datetime] = None
        self.break_task: Optional[asyncio.Task] = None
        self.completed = False
    
    def get_duration(self) -> float:
        """Get actual session duration in minutes"""
        end = self.actual_end_time or datetime.now()
        return (end - self.start_time).total_seconds() / 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "duration_minutes": self.duration_minutes,
            "actual_duration_minutes": self.get_duration(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "break_reminder": self.break_reminder,
            "completed": self.completed
        }


class FocusSkill(Skill):
    """
    Focus Mode skill for managing focus sessions and productivity.
    Supports Pomodoro-style sessions with break reminders.
    """
    
    def __init__(self):
        """Initialize focus skill"""
        super().__init__()
        self.name = "FocusSkill"
        self.description = "Manage focus sessions and productivity"
        self.permission_level = PermissionLevel.LOW
        self.reversible = True
        
        # Metadata
        self.metadata = SkillMetadata(
            name=self.name,
            description=self.description,
            category="productivity",
            tags=["focus", "productivity", "pomodoro", "concentration"],
            version="1.0.0",
            author="Porter.AI"
        )
        
        # Requirements
        self.requirements = SkillRequirements(
            min_confidence=0.7,
            required_intents=["start_focus", "end_focus", "focus_mode"],
            required_context_keys=[],
            cooldown_seconds=0
        )
        
        # Session tracking
        self.session_active = False
        self.current_session: Optional[FocusSession] = None
        self.session_history: List[FocusSession] = []
        self.break_reminder_triggered = False  # For testing
    
    async def validate(self, context: Dict[str, Any]) -> bool:
        """
        Validate focus skill can be executed.
        
        Args:
            context: Execution context
            
        Returns:
            True if valid
        """
        # Always allow focus mode operations
        return True
    
    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        """
        Execute focus action.
        
        Args:
            params: Execution parameters
                - action: "start", "end", "stats"
                - duration_minutes: Session duration (for start)
                - break_reminder: Enable break reminders (for start)
                
        Returns:
            SkillResult with session information
        """
        action = params.get("action", "start")
        
        try:
            if action == "start":
                return await self._start_session(params)
            elif action == "end":
                return await self._end_session()
            elif action == "stats":
                return await self._get_statistics()
            else:
                return SkillResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"Focus execution error: {e}")
            return SkillResult(
                success=False,
                error=str(e)
            )
    
    async def _start_session(self, params: Dict[str, Any]) -> SkillResult:
        """Start a focus session"""
        if self.session_active:
            return SkillResult(
                success=False,
                error="A focus session is already active"
            )
        
        duration = params.get("duration_minutes", 25)  # Default Pomodoro
        break_reminder = params.get("break_reminder", True)
        
        # Create session
        session = FocusSession(duration, break_reminder)
        self.current_session = session
        self.session_active = True
        
        # Start break reminder if enabled
        if break_reminder:
            session.break_task = asyncio.create_task(
                self._schedule_break_reminder(session)
            )
        
        logger.info(f"Started focus session: {duration} minutes")
        
        return SkillResult(
            success=True,
            data={
                "session_id": id(session),
                "duration_minutes": duration,
                "break_reminder": break_reminder,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat()
            },
            message=f"Focus session started for {duration} minutes",
            rollback_data=session
        )
    
    async def _end_session(self) -> SkillResult:
        """End the current focus session"""
        if not self.session_active or not self.current_session:
            return SkillResult(
                success=False,
                error="No active focus session"
            )
        
        session = self.current_session
        session.actual_end_time = datetime.now()
        session.completed = True
        
        # Cancel break reminder if active
        if session.break_task and not session.break_task.done():
            session.break_task.cancel()
        
        # Add to history
        self.session_history.append(session)
        
        # Calculate duration
        duration = session.get_duration()
        
        # Clear current session
        self.session_active = False
        self.current_session = None
        
        logger.info(f"Ended focus session: {duration:.1f} minutes")
        
        return SkillResult(
            success=True,
            data={
                "duration": duration,
                "planned_duration": session.duration_minutes,
                "completed": session.completed
            },
            message=f"Focus session ended after {duration:.1f} minutes"
        )
    
    async def _get_statistics(self) -> SkillResult:
        """Get focus session statistics"""
        total_sessions = len(self.session_history)
        
        if total_sessions == 0:
            return SkillResult(
                success=True,
                data={
                    "total_sessions": 0,
                    "total_focus_time": 0,
                    "average_session_length": 0
                },
                message="No focus sessions completed yet"
            )
        
        total_time = sum(s.get_duration() for s in self.session_history)
        avg_time = total_time / total_sessions
        
        # Get today's sessions
        today = datetime.now().date()
        today_sessions = [
            s for s in self.session_history
            if s.start_time.date() == today
        ]
        today_time = sum(s.get_duration() for s in today_sessions)
        
        return SkillResult(
            success=True,
            data={
                "total_sessions": total_sessions,
                "total_focus_time": total_time,
                "average_session_length": avg_time,
                "today_sessions": len(today_sessions),
                "today_focus_time": today_time
            },
            message=f"Completed {total_sessions} sessions, {total_time:.1f} minutes total"
        )
    
    async def _schedule_break_reminder(self, session: FocusSession) -> None:
        """Schedule a break reminder"""
        try:
            # Wait for session duration
            await asyncio.sleep(session.duration_minutes * 60)
            
            # Trigger break reminder
            await self._trigger_break_reminder()
            
        except asyncio.CancelledError:
            logger.debug("Break reminder cancelled")
        except Exception as e:
            logger.error(f"Break reminder error: {e}")
    
    async def _trigger_break_reminder(self) -> None:
        """Trigger break reminder notification"""
        self.break_reminder_triggered = True
        
        # Log reminder (in real app, would show system notification)
        logger.info("🔔 BREAK REMINDER: Time to take a break!")
        
        # In a real implementation:
        # - Show system notification
        # - Play a gentle sound
        # - Possibly pause active applications
    
    async def rollback(self) -> bool:
        """
        Rollback focus session start.
        
        Returns:
            True if successful
        """
        if self.session_active and self.current_session:
            # End the session
            await self._end_session()
        
        return True
    
    def cleanup(self) -> None:
        """Clean up active session"""
        if self.current_session and self.current_session.break_task:
            if not self.current_session.break_task.done():
                self.current_session.break_task.cancel()
        
        self.session_active = False
        self.current_session = None
        logger.info("Cleaned up focus session")