#!/usr/bin/env python3
"""
Activity Logger Skill for Porter.AI.
Logs activities and provides activity history and summaries.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import Counter
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory.enhanced_memory import EnhancedMemoryLayer
from ..core.skill_base import Skill, SkillResult, PermissionLevel, SkillMetadata, SkillRequirements

logger = logging.getLogger(__name__)


class LoggerSkill(Skill):
    """
    Activity Logger skill for tracking and querying activities.
    Uses enhanced memory layer for semantic search capabilities.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize logger skill.
        
        Args:
            persist_directory: Directory for persistent storage
        """
        super().__init__()
        self.name = "LoggerSkill"
        self.description = "Log and track activities"
        self.permission_level = PermissionLevel.LOW
        self.reversible = False  # Logging is append-only
        
        # Metadata
        self.metadata = SkillMetadata(
            name=self.name,
            description=self.description,
            category="productivity",
            tags=["logging", "tracking", "history", "analytics"],
            version="1.0.0",
            author="Porter.AI"
        )
        
        # Requirements
        self.requirements = SkillRequirements(
            min_confidence=0.6,
            required_intents=["log_activity", "query_activity", "activity_summary"],
            required_context_keys=[],
            cooldown_seconds=0
        )
        
        # Initialize memory layer for activity storage
        self.memory = EnhancedMemoryLayer(
            persist_directory=persist_directory or "./activity_logs",
            collection_name="porter_activities"
        )
        
        self._last_activity_id: Optional[str] = None
    
    async def validate(self, context: Dict[str, Any]) -> bool:
        """
        Validate logger skill can be executed.
        
        Args:
            context: Execution context
            
        Returns:
            True if valid
        """
        # Always allow logging operations
        return True
    
    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        """
        Execute logging action.
        
        Args:
            params: Execution parameters
                - action: "log" (default), "query", "summary", "clear"
                - activity: Activity type (for log)
                - description: Activity description (for log)
                - application: Application name (for log)
                - tags: List of tags (for log)
                - query: Search query (for query)
                - period: Time period (for summary)
                - confirm: Confirmation flag (for clear)
                
        Returns:
            SkillResult with activity information
        """
        action = params.get("action", "log")
        
        # Default to log if activity is provided
        if "activity" in params and action == "log":
            action = "log"
        
        try:
            if action == "log":
                return await self._log_activity(params)
            elif action == "query":
                return await self._query_activities(params)
            elif action == "summary":
                return await self._get_summary(params)
            elif action == "clear":
                return await self._clear_activities(params)
            else:
                return SkillResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"Logger execution error: {e}")
            return SkillResult(
                success=False,
                error=str(e)
            )
    
    async def _log_activity(self, params: Dict[str, Any]) -> SkillResult:
        """Log an activity"""
        activity = params.get("activity", "unknown")
        description = params.get("description", "")
        application = params.get("application", "")
        tags = params.get("tags", [])
        
        # Create context for memory storage
        context = {
            "activity": activity,
            "description": description,
            "application": application,
            "tags": tags,
            "timestamp": datetime.now().isoformat(),
            "importance": 0.5  # Default importance
        }
        
        # Store in memory
        activity_id = await self.memory.store_enhanced_context(context)
        self._last_activity_id = activity_id
        
        logger.info(f"Logged activity: {activity} - {description}")
        
        return SkillResult(
            success=True,
            data={"activity_id": activity_id},
            message=f"Activity logged: {activity}"
        )
    
    async def _query_activities(self, params: Dict[str, Any]) -> SkillResult:
        """Query logged activities"""
        query = params.get("query", "")
        limit = params.get("limit", 10)
        
        if not query:
            # Return recent activities
            activities = await self.memory.get_recent_contexts(limit)
        else:
            # Semantic search
            activities = await self.memory.search_similar_weighted(
                query=query,
                n_results=limit,
                use_temporal_weight=True
            )
        
        # Format activities
        formatted = []
        for act in activities:
            formatted.append({
                "activity": act.get("activity", "unknown"),
                "description": act.get("description", ""),
                "application": act.get("application", ""),
                "timestamp": act.get("timestamp", ""),
                "tags": act.get("tags", [])
            })
        
        return SkillResult(
            success=True,
            data={"activities": formatted},
            message=f"Found {len(formatted)} activities"
        )
    
    async def _get_summary(self, params: Dict[str, Any]) -> SkillResult:
        """Get activity summary"""
        period = params.get("period", "day")
        
        # Calculate time range
        now = datetime.now()
        if period == "day":
            start_time = now - timedelta(days=1)
        elif period == "week":
            start_time = now - timedelta(weeks=1)
        elif period == "month":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)
        
        # Query activities in time range
        activities = await self.memory.query_time_range(start_time, now)
        
        # Calculate statistics
        total = len(activities)
        
        # Count by activity type
        activity_counter = Counter()
        app_counter = Counter()
        
        for act in activities:
            activity_counter[act.get("activity", "unknown")] += 1
            if act.get("application"):
                app_counter[act.get("application")] += 1
        
        # Calculate time spent (simplified - assumes each activity is ~5 minutes)
        estimated_time = total * 5  # minutes
        
        return SkillResult(
            success=True,
            data={
                "total_activities": total,
                "period": period,
                "activity_breakdown": dict(activity_counter),
                "application_breakdown": dict(app_counter),
                "estimated_time_minutes": estimated_time,
                "most_common_activity": activity_counter.most_common(1)[0] if activity_counter else None
            },
            message=f"Summary for {period}: {total} activities logged"
        )
    
    async def _clear_activities(self, params: Dict[str, Any]) -> SkillResult:
        """Clear activity history"""
        if not params.get("confirm", False):
            return SkillResult(
                success=False,
                error="Confirmation required to clear activities"
            )
        
        # Clear memory collection
        # Note: Enhanced memory doesn't have a clear method, so we reinitialize
        self.memory._reinitialize_collection()
        
        logger.info("Cleared all activity history")
        
        return SkillResult(
            success=True,
            message="Activity history cleared"
        )
    
    async def rollback(self) -> bool:
        """
        Rollback not supported for logging.
        
        Returns:
            False
        """
        # Logging is append-only, no rollback
        return False
    
    def cleanup(self) -> None:
        """Clean up resources"""
        # Memory layer handles its own cleanup
        logger.info("Logger skill cleaned up")