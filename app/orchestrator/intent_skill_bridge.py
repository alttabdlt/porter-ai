#!/usr/bin/env python3
"""
Intent-Skills Bridge for Porter.AI.
Maps intents to appropriate skills and manages execution flow.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..intent.intent_router import IntentType
from ..skills.core.skill_base import Skill, SkillResult
from ..skills.core.skill_registry import SkillRegistry
from ..skills.core.skill_executor import SkillExecutor
from ..skills.core.permission_manager import PermissionManager

# Import core skills
from ..skills.implementations.timer_skill import TimerSkill
from ..skills.implementations.focus_skill import FocusSkill
from ..skills.implementations.logger_skill import LoggerSkill
from ..skills.implementations.status_skill import StatusSkill

logger = logging.getLogger(__name__)


@dataclass
class SkillMapping:
    """Maps intent types to skills"""
    intent_type: IntentType
    skill_name: str
    param_extractor: Optional[callable] = None


class IntentSkillBridge:
    """
    Bridges intent classification with skill execution.
    Maps intents to appropriate skills and manages the execution flow.
    """
    
    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        executor: Optional[SkillExecutor] = None,
        permission_manager: Optional[PermissionManager] = None
    ):
        """
        Initialize the bridge.
        
        Args:
            registry: Skill registry (creates new if None)
            executor: Skill executor (creates new if None)
            permission_manager: Permission manager (creates new if None)
        """
        self.registry = registry or SkillRegistry()
        self.permission_manager = permission_manager or PermissionManager()
        self.executor = executor or SkillExecutor(self.permission_manager)
        
        # Initialize skill mappings
        self.mappings: Dict[IntentType, List[SkillMapping]] = {}
        
        # Register core skills
        self._register_core_skills()
        
        # Set up intent mappings
        self._setup_mappings()
        
        logger.info("Intent-Skills bridge initialized")
    
    def _register_core_skills(self):
        """Register core safe skills"""
        core_skills = [
            TimerSkill(),
            FocusSkill(),
            LoggerSkill(),
            StatusSkill()
        ]
        
        for skill in core_skills:
            self.registry.register(skill)
            logger.info(f"Registered skill: {skill.name}")
    
    def _setup_mappings(self):
        """Set up intent to skill mappings"""
        # Timer/Reminder mappings
        self.add_mapping(
            IntentType.AUTOMATION,
            "TimerSkill",
            self._extract_timer_params
        )
        
        # Focus mode mappings
        self.add_mapping(
            IntentType.PRODUCTIVITY,
            "FocusSkill",
            self._extract_focus_params
        )
        
        # Activity logging mappings
        self.add_mapping(
            IntentType.NOTE_TAKING,
            "LoggerSkill",
            self._extract_logger_params
        )
        
        # Status check mappings
        self.add_mapping(
            IntentType.NAVIGATION,  # General queries
            "StatusSkill",
            self._extract_status_params
        )
        
        # Additional mappings for debugging
        self.add_mapping(
            IntentType.DEBUGGING,
            "StatusSkill",
            lambda c: {"type": "porter"}
        )
    
    def add_mapping(
        self,
        intent_type: IntentType,
        skill_name: str,
        param_extractor: Optional[callable] = None
    ):
        """
        Add an intent to skill mapping.
        
        Args:
            intent_type: Type of intent
            skill_name: Name of skill to execute
            param_extractor: Function to extract params from context
        """
        if intent_type not in self.mappings:
            self.mappings[intent_type] = []
        
        mapping = SkillMapping(intent_type, skill_name, param_extractor)
        self.mappings[intent_type].append(mapping)
        
        logger.debug(f"Added mapping: {intent_type.value} -> {skill_name}")
    
    async def route_intent(
        self,
        intent_type: IntentType,
        context: Dict[str, Any]
    ) -> List[SkillResult]:
        """
        Route an intent to appropriate skills.
        
        Args:
            intent_type: Classified intent type
            context: Execution context with details
            
        Returns:
            List of skill execution results
        """
        results = []
        
        # Get mappings for this intent
        mappings = self.mappings.get(intent_type, [])
        
        if not mappings:
            logger.warning(f"No skill mapping for intent: {intent_type.value}")
            return results
        
        # Execute each mapped skill
        for mapping in mappings:
            skill = self.registry.get(mapping.skill_name)
            
            if not skill:
                logger.error(f"Skill not found: {mapping.skill_name}")
                continue
            
            # Extract parameters
            if mapping.param_extractor:
                params = mapping.param_extractor(context)
            else:
                params = self._extract_default_params(context)
            
            # Add context to params
            params['_context'] = context
            
            # Execute skill
            try:
                result = await self.executor.execute_safe(
                    skill,
                    params,
                    context
                )
                results.append(result)
                
                logger.info(f"Executed {mapping.skill_name}: {result.success}")
                
            except Exception as e:
                logger.error(f"Error executing {mapping.skill_name}: {e}")
                results.append(SkillResult(
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    async def execute_skill_directly(
        self,
        skill_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SkillResult:
        """
        Execute a skill directly by name.
        
        Args:
            skill_name: Name of skill to execute
            params: Execution parameters
            context: Optional context
            
        Returns:
            Skill execution result
        """
        skill = self.registry.get(skill_name)
        
        if not skill:
            return SkillResult(
                success=False,
                error=f"Skill not found: {skill_name}"
            )
        
        context = context or {}
        
        try:
            result = await self.executor.execute_safe(
                skill,
                params,
                context
            )
            return result
            
        except Exception as e:
            logger.error(f"Error executing {skill_name}: {e}")
            return SkillResult(
                success=False,
                error=str(e)
            )
    
    def get_skills_for_intent(self, intent_type: IntentType) -> List[str]:
        """
        Get list of skills that handle an intent type.
        
        Args:
            intent_type: Intent type to query
            
        Returns:
            List of skill names
        """
        mappings = self.mappings.get(intent_type, [])
        return [m.skill_name for m in mappings]
    
    def get_all_mappings(self) -> Dict[str, List[str]]:
        """
        Get all intent to skill mappings.
        
        Returns:
            Dictionary of intent -> skill names
        """
        result = {}
        for intent_type, mappings in self.mappings.items():
            result[intent_type.value] = [m.skill_name for m in mappings]
        return result
    
    # Parameter extraction functions
    
    def _extract_default_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract default parameters from context"""
        return {
            "description": context.get("description", ""),
            "application": context.get("application", ""),
            "window_title": context.get("window_title", "")
        }
    
    def _extract_timer_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract timer parameters from context"""
        # Look for time-related keywords
        description = context.get("description", "").lower()
        
        # Simple duration extraction (would be more sophisticated in production)
        duration = 300  # Default 5 minutes
        if "minute" in description:
            # Extract number before "minute"
            import re
            match = re.search(r'(\d+)\s*minute', description)
            if match:
                duration = int(match.group(1)) * 60
        elif "hour" in description:
            match = re.search(r'(\d+)\s*hour', description)
            if match:
                duration = int(match.group(1)) * 3600
        
        return {
            "duration_seconds": duration,
            "message": context.get("description", "Timer completed!")
        }
    
    def _extract_focus_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract focus session parameters from context"""
        description = context.get("description", "").lower()
        
        # Check for focus actions
        if "end" in description or "stop" in description:
            return {"action": "end"}
        elif "stat" in description or "summary" in description:
            return {"action": "stats"}
        else:
            # Start new session
            duration = 25  # Default Pomodoro
            if "short" in description:
                duration = 15
            elif "long" in description:
                duration = 45
            
            return {
                "action": "start",
                "duration_minutes": duration,
                "break_reminder": True
            }
    
    def _extract_logger_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract logging parameters from context"""
        description = context.get("description", "").lower()
        
        # Check for logger actions
        if "search" in description or "find" in description or "query" in description:
            return {
                "action": "query",
                "query": context.get("description", "")
            }
        elif "summary" in description or "report" in description:
            period = "day"
            if "week" in description:
                period = "week"
            elif "month" in description:
                period = "month"
            return {
                "action": "summary",
                "period": period
            }
        else:
            # Default to logging
            return {
                "action": "log",
                "activity": context.get("activity", "unknown"),
                "description": context.get("description", ""),
                "application": context.get("application", ""),
                "tags": []
            }
    
    def _extract_status_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract status check parameters from context"""
        description = context.get("description", "").lower()
        
        # Determine status type
        if "skill" in description:
            return {"type": "skills"}
        elif "porter" in description or "component" in description:
            return {"type": "porter"}
        elif "performance" in description or "metric" in description:
            return {"type": "performance"}
        else:
            return {"type": "system"}  # Default to system status