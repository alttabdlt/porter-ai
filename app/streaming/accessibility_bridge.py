#!/usr/bin/env python3
"""
Accessibility API bridge for UI tree traversal and element detection.
Provides structured access to macOS UI elements and their relationships.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import Quartz
from ApplicationServices import (
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    AXUIElementCopyAttributeNames,
    AXUIElementGetPid,
    AXValueGetValue,
    kAXFocusedUIElementAttribute,
    kAXWindowsAttribute,
    kAXTitleAttribute,
    kAXRoleAttribute,
    kAXValueAttribute,
    kAXDescriptionAttribute,
    kAXPositionAttribute,
    kAXSizeAttribute,
    kAXChildrenAttribute,
    kAXFocusedAttribute,
    kAXEnabledAttribute
)
from Foundation import NSRunningApplication
from AppKit import NSWorkspace
import CoreFoundation as CF

logger = logging.getLogger(__name__)

@dataclass
class UIElement:
    """Represents a UI element with its properties"""
    role: str
    title: Optional[str] = None
    value: Optional[str] = None
    description: Optional[str] = None
    position: Optional[Tuple[float, float]] = None
    size: Optional[Tuple[float, float]] = None
    focused: bool = False
    enabled: bool = True
    pid: Optional[int] = None
    app_name: Optional[str] = None
    children: List['UIElement'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'role': self.role,
            'title': self.title,
            'value': self.value,
            'description': self.description,
            'position': self.position,
            'size': self.size,
            'focused': self.focused,
            'enabled': self.enabled,
            'app_name': self.app_name,
            'children_count': len(self.children)
        }

class AccessibilityBridge:
    """Bridge to macOS Accessibility API for UI inspection"""
    
    def __init__(self, cache_duration: float = 0.5):
        """
        Initialize Accessibility bridge.
        
        Args:
            cache_duration: How long to cache UI tree (seconds)
        """
        self.cache_duration = cache_duration
        self.last_tree: Optional[Dict] = None
        self.last_tree_time: float = 0
        self.focused_element: Optional[UIElement] = None
        
    def get_focused_element(self) -> Optional[UIElement]:
        """Get the currently focused UI element"""
        try:
            system_wide = AXUIElementCreateSystemWide()
            
            # Get focused element
            focused_ref = None
            result = AXUIElementCopyAttributeValue(
                system_wide,
                kAXFocusedUIElementAttribute,
                focused_ref
            )
            
            if result == 0 and focused_ref:
                element = self._parse_element(focused_ref[0])
                self.focused_element = element
                return element
                
        except Exception as e:
            logger.error(f"Error getting focused element: {e}")
        
        return None
    
    def get_frontmost_app(self) -> Optional[Dict]:
        """Get information about the frontmost application"""
        try:
            workspace = NSWorkspace.sharedWorkspace()
            frontmost = workspace.frontmostApplication()
            
            if frontmost:
                return {
                    'name': frontmost.localizedName(),
                    'bundle_id': frontmost.bundleIdentifier(),
                    'pid': frontmost.processIdentifier(),
                    'active': frontmost.isActive()
                }
                
        except Exception as e:
            logger.error(f"Error getting frontmost app: {e}")
        
        return None
    
    def get_ui_tree(self, max_depth: int = 5, force_refresh: bool = False) -> Dict:
        """
        Get the UI tree for the frontmost application.
        
        Args:
            max_depth: Maximum traversal depth
            force_refresh: Force refresh even if cached
            
        Returns:
            UI tree structure
        """
        # Check cache
        if not force_refresh and self.last_tree:
            if time.time() - self.last_tree_time < self.cache_duration:
                return self.last_tree
        
        start_time = time.time()
        tree = {
            'frontmost_app': self.get_frontmost_app(),
            'focused_element': None,
            'windows': [],
            'processing_time': 0
        }
        
        try:
            # Get focused element
            focused = self.get_focused_element()
            if focused:
                tree['focused_element'] = focused.to_dict()
            
            # Get frontmost app windows
            app_info = tree['frontmost_app']
            if app_info:
                windows = self._get_app_windows(app_info['pid'])
                tree['windows'] = [self._window_to_dict(w, max_depth) for w in windows]
            
        except Exception as e:
            logger.error(f"Error building UI tree: {e}")
        
        tree['processing_time'] = time.time() - start_time
        
        # Cache result
        self.last_tree = tree
        self.last_tree_time = time.time()
        
        logger.debug(f"UI tree built in {tree['processing_time']*1000:.1f}ms")
        
        return tree
    
    def _get_app_windows(self, pid: int) -> List:
        """Get windows for a specific application"""
        windows = []
        
        try:
            # Get app element
            app_ref = AXUIElementCreateApplication(pid)
            
            # Get windows attribute
            windows_ref = None
            result = AXUIElementCopyAttributeValue(
                app_ref,
                kAXWindowsAttribute,
                windows_ref
            )
            
            if result == 0 and windows_ref:
                # Parse windows
                for window_ref in windows_ref[0]:
                    windows.append(window_ref)
                    
        except Exception as e:
            logger.error(f"Error getting app windows: {e}")
        
        return windows
    
    def _window_to_dict(self, window_ref, max_depth: int) -> Dict:
        """Convert window reference to dictionary"""
        window = {
            'title': self._get_attribute(window_ref, kAXTitleAttribute),
            'role': self._get_attribute(window_ref, kAXRoleAttribute),
            'position': self._get_position(window_ref),
            'size': self._get_size(window_ref),
            'focused': self._get_attribute(window_ref, kAXFocusedAttribute, False),
            'elements': []
        }
        
        # Get child elements (limited depth)
        if max_depth > 0:
            children = self._get_attribute(window_ref, kAXChildrenAttribute, [])
            for child in children[:10]:  # Limit to first 10 children
                element = self._parse_element(child, max_depth - 1)
                if element:
                    window['elements'].append(element.to_dict())
        
        return window
    
    def _parse_element(self, element_ref, max_depth: int = 3) -> Optional[UIElement]:
        """Parse an accessibility element"""
        try:
            element = UIElement(
                role=self._get_attribute(element_ref, kAXRoleAttribute, "unknown")
            )
            
            # Get basic attributes
            element.title = self._get_attribute(element_ref, kAXTitleAttribute)
            element.value = self._get_attribute(element_ref, kAXValueAttribute)
            element.description = self._get_attribute(element_ref, kAXDescriptionAttribute)
            element.focused = self._get_attribute(element_ref, kAXFocusedAttribute, False)
            element.enabled = self._get_attribute(element_ref, kAXEnabledAttribute, True)
            
            # Get position and size
            element.position = self._get_position(element_ref)
            element.size = self._get_size(element_ref)
            
            # Get PID and app name
            pid = None
            result = AXUIElementGetPid(element_ref, pid)
            if result == 0 and pid:
                element.pid = pid[0]
                app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid[0])
                if app:
                    element.app_name = app.localizedName()
            
            # Get children (limited depth)
            if max_depth > 0:
                children = self._get_attribute(element_ref, kAXChildrenAttribute, [])
                for child in children[:5]:  # Limit children
                    child_element = self._parse_element(child, max_depth - 1)
                    if child_element:
                        element.children.append(child_element)
            
            return element
            
        except Exception as e:
            logger.debug(f"Error parsing element: {e}")
            return None
    
    def _get_attribute(self, element_ref, attribute, default=None):
        """Get an attribute value from an element"""
        try:
            value = None
            result = AXUIElementCopyAttributeValue(element_ref, attribute, value)
            if result == 0 and value:
                return value[0]
        except:
            pass
        return default
    
    def _get_position(self, element_ref) -> Optional[Tuple[float, float]]:
        """Get element position"""
        pos_value = self._get_attribute(element_ref, kAXPositionAttribute)
        if pos_value:
            point = CF.CFDictionaryCreateMutable(None, 0, None, None)
            if AXValueGetValue(pos_value, 1, point):  # kAXValueCGPointType = 1
                x = CF.CFDictionaryGetValue(point, "x")
                y = CF.CFDictionaryGetValue(point, "y")
                if x is not None and y is not None:
                    return (float(x), float(y))
        return None
    
    def _get_size(self, element_ref) -> Optional[Tuple[float, float]]:
        """Get element size"""
        size_value = self._get_attribute(element_ref, kAXSizeAttribute)
        if size_value:
            size = CF.CFDictionaryCreateMutable(None, 0, None, None)
            if AXValueGetValue(size_value, 2, size):  # kAXValueCGSizeType = 2
                width = CF.CFDictionaryGetValue(size, "width")
                height = CF.CFDictionaryGetValue(size, "height")
                if width is not None and height is not None:
                    return (float(width), float(height))
        return None
    
    def find_element_by_role(self, role: str) -> List[UIElement]:
        """Find all elements with a specific role"""
        elements = []
        
        def search(element: UIElement):
            if element.role == role:
                elements.append(element)
            for child in element.children:
                search(child)
        
        tree = self.get_ui_tree()
        for window in tree.get('windows', []):
            for element_dict in window.get('elements', []):
                # Would need to reconstruct UIElement from dict
                pass
        
        return elements
    
    def get_clickable_elements(self) -> List[Dict]:
        """Get all clickable elements in the current UI"""
        clickable_roles = ['AXButton', 'AXLink', 'AXMenuItem', 'AXTab']
        clickable = []
        
        tree = self.get_ui_tree()
        
        def find_clickable(elements: List[Dict], parent_name: str = ""):
            for elem in elements:
                if elem.get('role') in clickable_roles and elem.get('enabled', True):
                    clickable.append({
                        'title': elem.get('title', ''),
                        'role': elem.get('role'),
                        'position': elem.get('position'),
                        'parent': parent_name
                    })
                
                # Recurse into children
                if 'elements' in elem:
                    find_clickable(elem['elements'], elem.get('title', parent_name))
        
        for window in tree.get('windows', []):
            find_clickable(window.get('elements', []), window.get('title', ''))
        
        return clickable
    
    def get_text_fields(self) -> List[Dict]:
        """Get all text input fields"""
        text_roles = ['AXTextField', 'AXTextArea', 'AXSearchField', 'AXComboBox']
        fields = []
        
        tree = self.get_ui_tree()
        
        def find_fields(elements: List[Dict]):
            for elem in elements:
                if elem.get('role') in text_roles:
                    fields.append({
                        'title': elem.get('title', ''),
                        'value': elem.get('value', ''),
                        'role': elem.get('role'),
                        'position': elem.get('position'),
                        'focused': elem.get('focused', False)
                    })
                
                if 'elements' in elem:
                    find_fields(elem['elements'])
        
        for window in tree.get('windows', []):
            find_fields(window.get('elements', []))
        
        return fields
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        return {
            'cache_duration': self.cache_duration,
            'has_cached_tree': self.last_tree is not None,
            'cache_age': time.time() - self.last_tree_time if self.last_tree else None,
            'focused_element': self.focused_element.role if self.focused_element else None,
            'last_processing_time': self.last_tree.get('processing_time', 0) if self.last_tree else 0
        }


# Helper function for AXUIElementCreateApplication
def AXUIElementCreateApplication(pid):
    """Create an accessibility element for an application"""
    from ApplicationServices import AXUIElementCreateApplication as _AXUIElementCreateApplication
    return _AXUIElementCreateApplication(pid)


# Demo usage
if __name__ == "__main__":
    def demo():
        """Demonstrate Accessibility API bridge"""
        print("🔍 Accessibility API Demo")
        print("=" * 40)
        
        # Create bridge
        bridge = AccessibilityBridge()
        
        # Get frontmost app
        app = bridge.get_frontmost_app()
        if app:
            print(f"\n📱 Frontmost App:")
            print(f"  • Name: {app['name']}")
            print(f"  • Bundle ID: {app['bundle_id']}")
            print(f"  • PID: {app['pid']}")
        
        # Get focused element
        focused = bridge.get_focused_element()
        if focused:
            print(f"\n🎯 Focused Element:")
            print(f"  • Role: {focused.role}")
            print(f"  • Title: {focused.title}")
            print(f"  • Value: {focused.value}")
            print(f"  • Position: {focused.position}")
        
        # Get UI tree
        print("\n🌳 Building UI Tree...")
        tree = bridge.get_ui_tree(max_depth=3)
        
        print(f"\n📊 UI Tree Stats:")
        print(f"  • Windows: {len(tree.get('windows', []))}")
        print(f"  • Processing time: {tree.get('processing_time', 0)*1000:.1f}ms")
        
        # Get clickable elements
        clickable = bridge.get_clickable_elements()
        print(f"\n🖱️ Clickable Elements: {len(clickable)}")
        for elem in clickable[:5]:
            print(f"  • [{elem['role']}] {elem['title']}")
        
        # Get text fields
        fields = bridge.get_text_fields()
        print(f"\n📝 Text Fields: {len(fields)}")
        for field in fields[:5]:
            print(f"  • [{field['role']}] {field['title']} = '{field['value']}'")
        
        # Get metrics
        metrics = bridge.get_metrics()
        print(f"\n⚡ Performance Metrics:")
        for key, value in metrics.items():
            print(f"  • {key}: {value}")
    
    demo()