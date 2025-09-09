#!/usr/bin/env python3
"""
System Status Skill for Porter.AI.
Monitors system resources and Porter.AI component status.
"""

import logging
import psutil
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..core.skill_base import Skill, SkillResult, PermissionLevel, SkillMetadata, SkillRequirements
from ..core.skill_registry import SkillRegistry
from ..core.skill_executor import SkillExecutor

logger = logging.getLogger(__name__)


class StatusSkill(Skill):
    """
    System Status skill for monitoring system and Porter.AI status.
    Provides real-time information about resources and component health.
    """
    
    def __init__(self):
        """Initialize status skill"""
        super().__init__()
        self.name = "StatusSkill"
        self.description = "Monitor system and Porter.AI status"
        self.permission_level = PermissionLevel.LOW
        self.reversible = False  # Read-only operations
        
        # Metadata
        self.metadata = SkillMetadata(
            name=self.name,
            description=self.description,
            category="system",
            tags=["monitoring", "status", "health", "diagnostics"],
            version="1.0.0",
            author="Porter.AI"
        )
        
        # Requirements
        self.requirements = SkillRequirements(
            min_confidence=0.5,
            required_intents=["check_status", "system_info", "health_check"],
            required_context_keys=[],
            cooldown_seconds=0
        )
        
        # Performance tracking
        self._start_time = datetime.now()
        self._request_count = 0
        self._error_count = 0
        self._last_response_time = 0
    
    async def validate(self, context: Dict[str, Any]) -> bool:
        """
        Validate status skill can be executed.
        
        Args:
            context: Execution context
            
        Returns:
            True if valid
        """
        # Always allow status checks
        return True
    
    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        """
        Execute status check.
        
        Args:
            params: Execution parameters
                - type: "system", "porter", "skills", "performance"
                
        Returns:
            SkillResult with status information
        """
        status_type = params.get("type", "system")
        
        # Track request
        self._request_count += 1
        start = time.time()
        
        try:
            if status_type == "system":
                result = await self._get_system_status()
            elif status_type == "porter":
                result = await self._get_porter_status()
            elif status_type == "skills":
                result = await self._get_skills_status()
            elif status_type == "performance":
                result = await self._get_performance_metrics()
            else:
                result = SkillResult(
                    success=False,
                    error=f"Unknown status type: {status_type}"
                )
            
            # Track response time
            self._last_response_time = (time.time() - start) * 1000  # ms
            return result
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Status check error: {e}")
            return SkillResult(
                success=False,
                error=str(e)
            )
    
    async def _get_system_status(self) -> SkillResult:
        """Get system resource status"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network (if available)
            try:
                net_io = psutil.net_io_counters()
                network_sent_mb = net_io.bytes_sent / (1024**2)
                network_recv_mb = net_io.bytes_recv / (1024**2)
            except:
                network_sent_mb = 0
                network_recv_mb = 0
            
            return SkillResult(
                success=True,
                data={
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "memory_percent": memory_percent,
                    "memory_used_gb": round(memory_used_gb, 2),
                    "memory_total_gb": round(memory_total_gb, 2),
                    "disk_usage": disk_percent,
                    "disk_free_gb": round(disk_free_gb, 2),
                    "disk_total_gb": round(disk_total_gb, 2),
                    "network_sent_mb": round(network_sent_mb, 2),
                    "network_recv_mb": round(network_recv_mb, 2)
                },
                message=f"CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"
            )
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to get system status: {e}"
            )
    
    async def _get_porter_status(self) -> SkillResult:
        """Get Porter.AI component status"""
        components = {
            "perception": "active",  # Assume active if we're running
            "memory": "active",
            "intent": "active",
            "skills": "active",
            "safety": "active"
        }
        
        # Check component health (simplified)
        health_status = {
            "perception": self._check_perception_health(),
            "memory": self._check_memory_health(),
            "intent": self._check_intent_health(),
            "skills": self._check_skills_health(),
            "safety": self._check_safety_health()
        }
        
        # Overall health
        all_healthy = all(health_status.values())
        
        return SkillResult(
            success=True,
            data={
                "components": components,
                "health": health_status,
                "overall_health": "healthy" if all_healthy else "degraded",
                "uptime_hours": (datetime.now() - self._start_time).total_seconds() / 3600
            },
            message=f"Porter.AI {'healthy' if all_healthy else 'degraded'}"
        )
    
    async def _get_skills_status(self) -> SkillResult:
        """Get skills registry status"""
        try:
            # Create a registry to check
            registry = SkillRegistry()
            
            # Get registered skills (would be populated in real app)
            registered_skills = ["TimerSkill", "FocusSkill", "LoggerSkill", "StatusSkill"]
            
            # Get active skills (simplified - in real app, would check executor)
            active_skills = []  # Would get from skill executor
            
            return SkillResult(
                success=True,
                data={
                    "registered_skills": registered_skills,
                    "active_skills": active_skills,
                    "total_registered": len(registered_skills),
                    "total_active": len(active_skills)
                },
                message=f"{len(registered_skills)} skills registered"
            )
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to get skills status: {e}"
            )
    
    async def _get_performance_metrics(self) -> SkillResult:
        """Get performance metrics"""
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        # Calculate metrics
        avg_response_time = self._last_response_time  # Simplified
        throughput = self._request_count / uptime if uptime > 0 else 0
        error_rate = self._error_count / self._request_count if self._request_count > 0 else 0
        
        return SkillResult(
            success=True,
            data={
                "response_time_ms": round(avg_response_time, 2),
                "throughput": round(throughput, 2),
                "error_rate": round(error_rate * 100, 2),
                "total_requests": self._request_count,
                "total_errors": self._error_count,
                "uptime_seconds": round(uptime, 0)
            },
            message=f"Response time: {avg_response_time:.1f}ms, Error rate: {error_rate*100:.1f}%"
        )
    
    def _check_perception_health(self) -> bool:
        """Check perception layer health"""
        # In real app, would check if screen capture is working
        return True
    
    def _check_memory_health(self) -> bool:
        """Check memory layer health"""
        # In real app, would check ChromaDB connection
        return True
    
    def _check_intent_health(self) -> bool:
        """Check intent router health"""
        # In real app, would test intent classification
        return True
    
    def _check_skills_health(self) -> bool:
        """Check skills layer health"""
        # In real app, would check skill executor
        return True
    
    def _check_safety_health(self) -> bool:
        """Check safety layer health"""
        # In real app, would check permission manager
        return True
    
    async def rollback(self) -> bool:
        """
        Rollback not supported for status checks.
        
        Returns:
            False
        """
        return False
    
    def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Status skill cleaned up")