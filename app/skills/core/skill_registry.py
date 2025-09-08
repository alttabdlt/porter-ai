#!/usr/bin/env python3
"""
Skill Registry for discovering, registering, and managing skills.
"""

import logging
import importlib
import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Type

from .skill_base import Skill, SkillMetadata

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    Registry for managing available skills.
    Handles discovery, registration, and retrieval of skills.
    """
    
    def __init__(self):
        """Initialize empty skill registry"""
        self._skills: Dict[str, Skill] = {}
        self._categories: Dict[str, List[Skill]] = {}
        self._tags: Dict[str, List[Skill]] = {}
    
    def register(self, skill: Skill) -> None:
        """
        Register a skill instance.
        
        Args:
            skill: Skill instance to register
        """
        name = skill.name if hasattr(skill, 'name') else skill.__class__.__name__
        
        if name in self._skills:
            logger.warning(f"Overwriting existing skill: {name}")
        
        self._skills[name] = skill
        
        # Index by category
        if skill.metadata:
            category = skill.metadata.category
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(skill)
            
            # Index by tags
            for tag in skill.metadata.tags:
                if tag not in self._tags:
                    self._tags[tag] = []
                self._tags[tag].append(skill)
        
        logger.info(f"Registered skill: {name}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a skill.
        
        Args:
            name: Name of skill to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if name not in self._skills:
            return False
        
        skill = self._skills[name]
        
        # Remove from main registry
        del self._skills[name]
        
        # Remove from category index
        if skill.metadata and skill.metadata.category in self._categories:
            self._categories[skill.metadata.category].remove(skill)
            if not self._categories[skill.metadata.category]:
                del self._categories[skill.metadata.category]
        
        # Remove from tag index
        if skill.metadata:
            for tag in skill.metadata.tags:
                if tag in self._tags:
                    self._tags[tag].remove(skill)
                    if not self._tags[tag]:
                        del self._tags[tag]
        
        logger.info(f"Unregistered skill: {name}")
        return True
    
    def get(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.
        
        Args:
            name: Name of skill
            
        Returns:
            Skill instance or None
        """
        return self._skills.get(name)
    
    def get_all(self) -> List[Skill]:
        """Get all registered skills"""
        return list(self._skills.values())
    
    def get_by_category(self, category: str) -> List[Skill]:
        """
        Get skills by category.
        
        Args:
            category: Category name
            
        Returns:
            List of skills in category
        """
        return self._categories.get(category, [])
    
    def get_by_tag(self, tag: str) -> List[Skill]:
        """
        Get skills by tag.
        
        Args:
            tag: Tag name
            
        Returns:
            List of skills with tag
        """
        return self._tags.get(tag, [])
    
    def get_categories(self) -> List[str]:
        """Get all categories"""
        return list(self._categories.keys())
    
    def get_tags(self) -> List[str]:
        """Get all tags"""
        return list(self._tags.keys())
    
    def discover_skills(self, directory: str) -> List[str]:
        """
        Discover and register skills from a directory.
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of discovered skill names
        """
        discovered = []
        skills_dir = Path(directory)
        
        if not skills_dir.exists():
            logger.warning(f"Skills directory does not exist: {directory}")
            return discovered
        
        # Search for Python files
        for file_path in skills_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            module_name = file_path.stem
            
            try:
                # Import module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find Skill subclasses
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Skill) and 
                            obj != Skill and
                            not inspect.isabstract(obj)):
                            
                            # Instantiate and register
                            try:
                                skill_instance = obj()
                                self.register(skill_instance)
                                discovered.append(skill_instance.name)
                                logger.info(f"Discovered skill: {skill_instance.name}")
                            except Exception as e:
                                logger.error(f"Failed to instantiate skill {name}: {e}")
                
            except Exception as e:
                logger.error(f"Failed to import module {module_name}: {e}")
        
        logger.info(f"Discovered {len(discovered)} skills from {directory}")
        return discovered
    
    def search(self, query: str) -> List[Skill]:
        """
        Search for skills by name, category, or tag.
        
        Args:
            query: Search query
            
        Returns:
            List of matching skills
        """
        query_lower = query.lower()
        results = set()
        
        # Search by name
        for name, skill in self._skills.items():
            if query_lower in name.lower():
                results.add(skill)
        
        # Search by category
        for category, skills in self._categories.items():
            if query_lower in category.lower():
                results.update(skills)
        
        # Search by tag
        for tag, skills in self._tags.items():
            if query_lower in tag.lower():
                results.update(skills)
        
        # Search in descriptions
        for skill in self._skills.values():
            if skill.metadata and query_lower in skill.metadata.description.lower():
                results.add(skill)
        
        return list(results)
    
    def clear(self) -> None:
        """Clear all registered skills"""
        self._skills.clear()
        self._categories.clear()
        self._tags.clear()
        logger.info("Cleared skill registry")
    
    def __len__(self) -> int:
        """Get number of registered skills"""
        return len(self._skills)
    
    def __contains__(self, name: str) -> bool:
        """Check if skill is registered"""
        return name in self._skills
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"<SkillRegistry: {len(self._skills)} skills, {len(self._categories)} categories>"