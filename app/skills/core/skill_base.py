#!/usr/bin/env python3
"""
Base Skill class and core data structures for the Skills Layer.
Provides safe, reversible action execution with permission management.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for skill execution"""
    LOW = "low"       # Safe, read-only or minimal impact
    MEDIUM = "medium" # Moderate impact, may change state
    HIGH = "high"     # High impact, potentially irreversible


@dataclass
class SkillResult:
    """Result of skill execution"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    rollback_data: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SkillMetadata:
    """Metadata for a skill"""
    name: str
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = ""


@dataclass
class SkillRequirements:
    """Requirements for skill execution"""
    min_confidence: float = 0.5
    required_intents: List[str] = field(default_factory=list)
    required_context_keys: List[str] = field(default_factory=list)
    cooldown_seconds: int = 0


class Skill(ABC):
    """
    Abstract base class for all skills.
    Defines the interface and lifecycle for safe skill execution.
    """
    
    def __init__(self):
        """Initialize skill with default properties"""
        self.name = self.__class__.__name__
        self.description = ""
        self.permission_level = PermissionLevel.LOW
        self.reversible = True
        self.requires_confirmation = False
        self.max_execution_time = 30  # seconds
        self.metadata: Optional[SkillMetadata] = None
        self.requirements: Optional[SkillRequirements] = None
        self._last_result: Optional[SkillResult] = None
    
    @abstractmethod
    async def validate(self, context: Dict[str, Any]) -> bool:
        """
        Validate that the skill can be executed in the current context.
        
        Args:
            context: Current execution context
            
        Returns:
            True if skill can be executed, False otherwise
        """
        pass
    
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        """
        Execute the skill with given parameters.
        
        Args:
            params: Execution parameters
            
        Returns:
            SkillResult with execution outcome
        """
        pass
    
    @abstractmethod
    async def rollback(self) -> bool:
        """
        Rollback the skill's last execution if possible.
        
        Returns:
            True if rollback successful, False otherwise
        """
        pass
    
    def get_description(self) -> str:
        """Get human-readable skill description"""
        if self.metadata:
            return self.metadata.description
        return self.description or f"{self.name} skill"
    
    def get_permission_level(self) -> PermissionLevel:
        """Get required permission level"""
        return self.permission_level
    
    def is_reversible(self) -> bool:
        """Check if skill can be rolled back"""
        return self.reversible
    
    def needs_confirmation(self) -> bool:
        """Check if skill requires user confirmation"""
        return self.requires_confirmation or self.permission_level == PermissionLevel.HIGH
    
    def check_requirements(self, context: Dict[str, Any]) -> bool:
        """
        Check if skill requirements are met.
        
        Args:
            context: Current context
            
        Returns:
            True if requirements met
        """
        if not self.requirements:
            return True
        
        # Check confidence
        confidence = context.get('confidence', 0)
        if confidence < self.requirements.min_confidence:
            logger.debug(f"Confidence {confidence} below minimum {self.requirements.min_confidence}")
            return False
        
        # Check required intents
        if self.requirements.required_intents:
            intent = context.get('intent', '')
            if intent not in self.requirements.required_intents:
                logger.debug(f"Intent {intent} not in required {self.requirements.required_intents}")
                return False
        
        # Check required context keys
        for key in self.requirements.required_context_keys:
            if key not in context:
                logger.debug(f"Missing required context key: {key}")
                return False
        
        return True
    
    async def pre_execute(self, params: Dict[str, Any]) -> bool:
        """
        Hook called before execution.
        Override in subclasses for setup.
        
        Args:
            params: Execution parameters
            
        Returns:
            True to proceed with execution
        """
        return True
    
    async def post_execute(self, result: SkillResult) -> None:
        """
        Hook called after execution.
        Override in subclasses for cleanup.
        
        Args:
            result: Execution result
        """
        self._last_result = result
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name} ({self.permission_level.value})"
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"<Skill: {self.name} perm={self.permission_level.value} reversible={self.reversible}>"