#!/usr/bin/env python3
"""
Memory safety configuration for FASTVLM Jarvis
Prevents crashes and manages resources
"""

class MemorySafetyConfig:
    """Configuration for memory-safe operation"""
    
    # Memory Limits (MB)
    MAX_BACKEND_MEMORY = 500  # Maximum backend memory before aggressive cleanup
    MAX_FRONTEND_MEMORY = 50  # Maximum frontend screenshot cache
    WARNING_THRESHOLD = 300  # Warning when growth exceeds this
    CRITICAL_THRESHOLD = 500  # Critical actions when growth exceeds this
    
    # Screenshot Management
    MAX_SCREENSHOT_SIZE_KB = 500  # Maximum size per screenshot
    SCREENSHOT_QUALITY = 70  # JPEG quality (lower = smaller)
    RESIZE_FACTOR = 0.5  # Resize screenshots to 50% for storage
    
    # Buffer Management
    MAX_BUFFER_FRAMES = 150  # Maximum frames in buffer (15 fps * 10 seconds)
    BUFFER_CLEANUP_INTERVAL = 30  # Clean buffer every N seconds
    
    # Event Management
    MAX_EVENTS_IN_MEMORY = 100  # Maximum events to keep
    EVENT_CLEANUP_BATCH = 20  # Remove this many events when cleaning
    
    # Garbage Collection
    GC_INTERVAL_EVENTS = 50  # Force GC every N events
    GC_INTERVAL_SECONDS = 60  # Force GC every N seconds
    
    # Safety Features
    AUTO_RESTART_ON_CRITICAL = True  # Restart if memory critical
    PAUSE_ON_HIGH_MEMORY = True  # Pause capture when memory high
    LOG_MEMORY_STATS = True  # Log detailed memory stats
    
    @staticmethod
    def get_safe_settings():
        """Get settings for safe operation"""
        return {
            'capture': {
                'fps': 5,  # Lower FPS to reduce memory
                'buffer_seconds': 10,  # Shorter buffer
                'resize_captures': True,
                'compression': 'high'
            },
            'vlm': {
                'batch_size': 1,  # Process one at a time
                'cache_results': False,  # Don't cache VLM results
                'simplified_mode': True  # Use simplified VLM
            },
            'dashboard': {
                'max_events': 50,  # Show only recent 50
                'auto_cleanup': True,
                'cleanup_interval': 30
            }
        }
    
    @staticmethod
    def calculate_memory_budget():
        """Calculate memory budget based on system"""
        import psutil
        
        total_memory = psutil.virtual_memory().total / 1024 / 1024  # MB
        available = psutil.virtual_memory().available / 1024 / 1024  # MB
        
        # Use at most 10% of system memory or 1GB, whichever is less
        max_allowed = min(total_memory * 0.1, 1024)
        
        # But don't exceed available - 500MB (safety margin)
        safe_limit = min(max_allowed, available - 500)
        
        return {
            'total_system_mb': total_memory,
            'available_mb': available,
            'recommended_limit_mb': safe_limit,
            'warning_at_mb': safe_limit * 0.7,
            'critical_at_mb': safe_limit * 0.9
        }

class MemoryMonitor:
    """Advanced memory monitoring with crash prevention"""
    
    def __init__(self, config=None):
        self.config = config or MemorySafetyConfig()
        self.measurements = []
        self.alerts = []
        
    def check_memory_health(self, current_mb, growth_mb, events):
        """Check memory health and return actions"""
        actions = []
        
        # Check growth rate
        if events > 0:
            mb_per_event = growth_mb / events
            if mb_per_event > 10:
                actions.append({
                    'type': 'warning',
                    'message': f'High memory per event: {mb_per_event:.1f}MB',
                    'action': 'reduce_screenshot_quality'
                })
        
        # Check absolute growth
        if growth_mb > self.config.WARNING_THRESHOLD:
            actions.append({
                'type': 'warning',
                'message': f'Memory growth warning: {growth_mb:.1f}MB',
                'action': 'force_cleanup'
            })
            
        if growth_mb > self.config.CRITICAL_THRESHOLD:
            actions.append({
                'type': 'critical',
                'message': f'Memory critical: {growth_mb:.1f}MB',
                'action': 'emergency_cleanup'
            })
            
        # Check system memory
        import psutil
        system_mem = psutil.virtual_memory()
        if system_mem.percent > 80:
            actions.append({
                'type': 'critical',
                'message': f'System memory critical: {system_mem.percent}%',
                'action': 'pause_capture'
            })
            
        return actions
    
    def get_memory_report(self):
        """Generate detailed memory report"""
        import psutil
        import gc
        
        # Get memory stats
        process = psutil.Process()
        mem_info = process.memory_info()
        
        # Get object counts
        obj_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            obj_counts[obj_type] = obj_counts.get(obj_type, 0) + 1
            
        # Sort by count
        top_objects = sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'process_memory_mb': mem_info.rss / 1024 / 1024,
            'virtual_memory_mb': mem_info.vms / 1024 / 1024,
            'system_memory_percent': psutil.virtual_memory().percent,
            'top_objects': top_objects,
            'gc_stats': gc.get_stats(),
            'recommendations': self._get_recommendations(mem_info.rss / 1024 / 1024)
        }
    
    def _get_recommendations(self, current_mb):
        """Get recommendations based on memory usage"""
        recs = []
        
        if current_mb > 1000:
            recs.append("Reduce capture FPS to 2-5")
            recs.append("Disable screenshot capture temporarily")
            recs.append("Reduce buffer size to 5 seconds")
            
        elif current_mb > 500:
            recs.append("Reduce screenshot quality to 50%")
            recs.append("Limit buffer to 10 seconds")
            recs.append("Increase GC frequency")
            
        elif current_mb > 300:
            recs.append("Monitor memory growth closely")
            recs.append("Consider reducing FPS")
            
        return recs

def apply_memory_safety(jarvis_system):
    """Apply memory safety settings to Jarvis system"""
    config = MemorySafetyConfig()
    
    # Reduce capture settings
    jarvis_system.capture.target_fps = 5
    jarvis_system.capture.buffer_seconds = 10
    
    # Set analysis interval
    jarvis_system.analysis_interval = 3.0
    
    # Set GC interval
    jarvis_system.gc_interval = config.GC_INTERVAL_EVENTS
    
    # Add memory monitor
    jarvis_system.memory_monitor = MemoryMonitor(config)
    
    return jarvis_system

# Quick test
if __name__ == "__main__":
    config = MemorySafetyConfig()
    budget = config.calculate_memory_budget()
    
    print("Memory Safety Configuration")
    print("="*50)
    print(f"System Memory: {budget['total_system_mb']:.0f}MB")
    print(f"Available: {budget['available_mb']:.0f}MB")
    print(f"Recommended Limit: {budget['recommended_limit_mb']:.0f}MB")
    print(f"Warning at: {budget['warning_at_mb']:.0f}MB")
    print(f"Critical at: {budget['critical_at_mb']:.0f}MB")
    print()
    print("Safe Settings:")
    for key, value in config.get_safe_settings().items():
        print(f"  {key}: {value}")