#!/usr/bin/env python3
"""
Permission Manager for handling user consent and skill permissions.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Set

from .skill_base import Skill, PermissionLevel

logger = logging.getLogger(__name__)


class PermissionManager:
    """
    Manages permissions and user consent for skill execution.
    """
    
    def __init__(self, permission_file: Optional[str] = None):
        """
        Initialize permission manager.
        
        Args:
            permission_file: Path to permissions storage file
        """
        self.permission_file = permission_file or os.path.expanduser("~/.porter_permissions.json")
        self._permissions: Dict[str, str] = {}  # skill_name -> permission_level
        self._temporary_permissions: Dict[str, datetime] = {}  # skill_name -> expiry
        self._denied_skills: Set[str] = set()
        self.auto_approve = False  # For testing
        
        # Load existing permissions
        self.load()
    
    def has_permission(self, skill_name: str, required_level: PermissionLevel) -> bool:
        """
        Check if skill has required permission level.
        
        Args:
            skill_name: Name of skill
            required_level: Required permission level
            
        Returns:
            True if permission granted
        """
        # Check if explicitly denied
        if skill_name in self._denied_skills:
            return False
        
        # Check temporary permissions
        if skill_name in self._temporary_permissions:
            if datetime.now() < self._temporary_permissions[skill_name]:
                # Still valid
                return True
            else:
                # Expired
                del self._temporary_permissions[skill_name]
        
        # Check permanent permissions
        if skill_name not in self._permissions:
            return False
        
        granted_level = self._permissions[skill_name]
        
        # Compare permission levels
        level_order = {
            PermissionLevel.LOW.value: 1,
            PermissionLevel.MEDIUM.value: 2,
            PermissionLevel.HIGH.value: 3
        }
        
        return level_order.get(granted_level, 0) >= level_order.get(required_level.value, 999)
    
    def set_permission(self, skill_name: str, level: PermissionLevel) -> None:
        """
        Set permanent permission for a skill.
        
        Args:
            skill_name: Name of skill
            level: Permission level to grant
        """
        self._permissions[skill_name] = level.value
        
        # Remove from denied list if present
        self._denied_skills.discard(skill_name)
        
        # Save changes
        self.save()
        
        logger.info(f"Set permission for {skill_name}: {level.value}")
    
    def grant_temporary(self, skill_name: str, duration_minutes: int = 60) -> None:
        """
        Grant temporary permission.
        
        Args:
            skill_name: Name of skill
            duration_minutes: Duration in minutes
        """
        expiry = datetime.now() + timedelta(minutes=duration_minutes)
        self._temporary_permissions[skill_name] = expiry
        
        logger.info(f"Granted temporary permission for {skill_name} until {expiry}")
    
    def deny(self, skill_name: str) -> None:
        """
        Explicitly deny a skill.
        
        Args:
            skill_name: Name of skill to deny
        """
        self._denied_skills.add(skill_name)
        
        # Remove any existing permissions
        self._permissions.pop(skill_name, None)
        self._temporary_permissions.pop(skill_name, None)
        
        # Save changes
        self.save()
        
        logger.info(f"Denied permission for {skill_name}")
    
    async def request_permission(self, skill: Skill) -> bool:
        """
        Request permission from user for skill execution.
        
        Args:
            skill: Skill requesting permission
            
        Returns:
            True if permission granted
        """
        # For testing - auto approve or deny
        if hasattr(self, 'auto_approve'):
            if self.auto_approve:
                self.set_permission(skill.name, skill.permission_level)
                return True
            else:
                return False
        
        # Check if already has permission
        if self.has_permission(skill.name, skill.permission_level):
            return True
        
        # In real implementation, this would show a UI prompt
        # For now, we'll check environment or return False
        if os.getenv("PORTER_AUTO_APPROVE") == "true":
            self.set_permission(skill.name, skill.permission_level)
            return True
        
        logger.warning(f"Permission required for {skill.name} ({skill.permission_level.value})")
        return False
    
    def revoke(self, skill_name: str) -> bool:
        """
        Revoke permissions for a skill.
        
        Args:
            skill_name: Name of skill
            
        Returns:
            True if revoked
        """
        changed = False
        
        if skill_name in self._permissions:
            del self._permissions[skill_name]
            changed = True
        
        if skill_name in self._temporary_permissions:
            del self._temporary_permissions[skill_name]
            changed = True
        
        if changed:
            self.save()
            logger.info(f"Revoked permissions for {skill_name}")
        
        return changed
    
    def clear_expired(self) -> int:
        """
        Clear expired temporary permissions.
        
        Returns:
            Number of expired permissions cleared
        """
        now = datetime.now()
        expired = [
            skill for skill, expiry in self._temporary_permissions.items()
            if expiry < now
        ]
        
        for skill in expired:
            del self._temporary_permissions[skill]
        
        if expired:
            logger.info(f"Cleared {len(expired)} expired permissions")
        
        return len(expired)
    
    def get_all_permissions(self) -> Dict[str, Dict]:
        """
        Get all current permissions.
        
        Returns:
            Dictionary of all permissions
        """
        self.clear_expired()
        
        return {
            "permanent": self._permissions.copy(),
            "temporary": {
                skill: expiry.isoformat()
                for skill, expiry in self._temporary_permissions.items()
            },
            "denied": list(self._denied_skills)
        }
    
    def save(self) -> None:
        """Save permissions to file"""
        try:
            data = {
                "permissions": self._permissions,
                "denied": list(self._denied_skills),
                "updated": datetime.now().isoformat()
            }
            
            Path(self.permission_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.permission_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved permissions to {self.permission_file}")
            
        except Exception as e:
            logger.error(f"Failed to save permissions: {e}")
    
    def load(self) -> None:
        """Load permissions from file"""
        try:
            if not os.path.exists(self.permission_file):
                logger.debug("No permission file found, starting fresh")
                return
            
            with open(self.permission_file, 'r') as f:
                data = json.load(f)
            
            self._permissions = data.get("permissions", {})
            self._denied_skills = set(data.get("denied", []))
            
            logger.info(f"Loaded {len(self._permissions)} permissions")
            
        except Exception as e:
            logger.error(f"Failed to load permissions: {e}")
    
    def reset(self) -> None:
        """Reset all permissions"""
        self._permissions.clear()
        self._temporary_permissions.clear()
        self._denied_skills.clear()
        self.save()
        logger.info("Reset all permissions")
    
    def __repr__(self) -> str:
        """Developer representation"""
        return (f"<PermissionManager: {len(self._permissions)} permanent, "
                f"{len(self._temporary_permissions)} temporary, "
                f"{len(self._denied_skills)} denied>")