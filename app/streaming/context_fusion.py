#!/usr/bin/env python3
"""
Context fusion system that combines multiple intelligence sources.
Merges VLM, OCR, UI tree, and saliency into unified understanding.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import numpy as np
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class FusedContext:
    """Unified context from all sources"""
    timestamp: float
    
    # Primary understanding
    description: str
    confidence: float
    importance: float
    
    # Source contributions
    vlm_output: Optional[str] = None
    ocr_text: List[str] = field(default_factory=list)
    ui_elements: List[Dict] = field(default_factory=list)
    
    # Derived insights
    application: Optional[str] = None
    activity_type: Optional[str] = None  # coding, browsing, writing, etc.
    key_entities: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    
    # Metadata
    processing_time: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    saliency_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'description': self.description,
            'confidence': self.confidence,
            'importance': self.importance,
            'application': self.application,
            'activity_type': self.activity_type,
            'key_entities': self.key_entities,
            'suggested_actions': self.suggested_actions,
            'sources_used': self.sources_used,
            'processing_time': self.processing_time
        }

class ContextFusion:
    """
    Fuses multiple intelligence sources into coherent context.
    Handles conflicts, weights confidence, and maintains temporal consistency.
    """
    
    def __init__(self, history_size: int = 30):
        """
        Initialize context fusion.
        
        Args:
            history_size: Number of contexts to keep in history
        """
        self.history_size = history_size
        self.context_history = deque(maxlen=history_size)
        
        # Activity patterns
        self.activity_patterns = {
            'coding': ['code', 'function', 'variable', 'class', 'import', 'terminal'],
            'browsing': ['browser', 'url', 'search', 'website', 'tab', 'link'],
            'writing': ['document', 'text', 'paragraph', 'email', 'note', 'typing'],
            'reading': ['article', 'pdf', 'book', 'scroll', 'page'],
            'designing': ['canvas', 'layer', 'shape', 'color', 'tool', 'brush'],
            'messaging': ['chat', 'message', 'conversation', 'reply', 'send']
        }
        
        # Confidence weights for different sources
        self.source_weights = {
            'vlm': 0.4,
            'ocr': 0.3,
            'ui_tree': 0.2,
            'saliency': 0.1
        }
        
        # Performance tracking
        self.fusion_times = deque(maxlen=100)
        
    def fuse(self,
            vlm_output: Optional[str] = None,
            ocr_result: Optional[Any] = None,  # OCRResult from vision_ocr
            ui_tree: Optional[Dict] = None,
            saliency_metrics: Optional[Any] = None,  # FrameMetrics from saliency_gate
            frame_timestamp: Optional[float] = None) -> FusedContext:
        """
        Fuse multiple intelligence sources into unified context.
        
        Args:
            vlm_output: Vision-language model output
            ocr_result: OCR detection result
            ui_tree: UI accessibility tree
            saliency_metrics: Frame saliency metrics
            frame_timestamp: Timestamp of the frame
            
        Returns:
            FusedContext with unified understanding
        """
        start_time = time.time()
        timestamp = frame_timestamp or time.time()
        
        # Initialize context with required fields
        context = FusedContext(
            timestamp=timestamp,
            description="",
            confidence=0.0,
            importance=0.0
        )
        sources_used = []
        
        # Extract information from each source
        vlm_info = self._process_vlm(vlm_output) if vlm_output else {}
        ocr_info = self._process_ocr(ocr_result) if ocr_result else {}
        ui_info = self._process_ui_tree(ui_tree) if ui_tree else {}
        saliency_info = self._process_saliency(saliency_metrics) if saliency_metrics else {}
        
        # Store source data
        if vlm_output:
            context.vlm_output = vlm_output
            sources_used.append('vlm')
        
        if ocr_info:
            context.ocr_text = ocr_info.get('text_lines', [])
            sources_used.append('ocr')
        
        if ui_info:
            context.ui_elements = ui_info.get('elements', [])
            context.application = ui_info.get('app_name')
            sources_used.append('ui_tree')
        
        if saliency_info:
            context.saliency_regions = saliency_info.get('regions', [])
            sources_used.append('saliency')
        
        context.sources_used = sources_used
        
        # Fuse information with confidence weighting
        description = self._generate_description(vlm_info, ocr_info, ui_info, saliency_info)
        context.description = description
        
        # Calculate confidence
        context.confidence = self._calculate_confidence(vlm_info, ocr_info, ui_info, saliency_info)
        
        # Calculate importance
        context.importance = self._calculate_importance(vlm_info, ocr_info, ui_info, saliency_info)
        
        # Detect activity type
        context.activity_type = self._detect_activity_type(description, ocr_info, ui_info)
        
        # Extract key entities
        context.key_entities = self._extract_entities(vlm_info, ocr_info, ui_info)
        
        # Generate suggested actions
        context.suggested_actions = self._suggest_actions(context)
        
        # Apply temporal smoothing
        context = self._apply_temporal_smoothing(context)
        
        # Track performance
        processing_time = time.time() - start_time
        context.processing_time = processing_time
        self.fusion_times.append(processing_time)
        
        # Add to history
        self.context_history.append(context)
        
        logger.debug(f"Context fusion complete in {processing_time*1000:.1f}ms using {sources_used}")
        
        return context
    
    def _process_vlm(self, vlm_output: str) -> Dict:
        """Process VLM output"""
        return {
            'description': vlm_output,
            'confidence': 0.8,  # Default confidence for VLM
            'entities': self._extract_entities_from_text(vlm_output)
        }
    
    def _process_ocr(self, ocr_result) -> Dict:
        """Process OCR result"""
        if not ocr_result or not hasattr(ocr_result, 'regions'):
            return {}
        
        text_lines = [r.text for r in ocr_result.regions if r.confidence > 0.5]
        high_conf_text = [r.text for r in ocr_result.regions if r.confidence > 0.8]
        
        return {
            'text_lines': text_lines,
            'high_conf_text': high_conf_text,
            'word_count': ocr_result.total_words,
            'avg_confidence': ocr_result.average_confidence,
            'entities': self._extract_entities_from_text(' '.join(text_lines))
        }
    
    def _process_ui_tree(self, ui_tree: Dict) -> Dict:
        """Process UI tree"""
        if not ui_tree:
            return {}
        
        app_info = ui_tree.get('frontmost_app', {})
        focused = ui_tree.get('focused_element', {})
        
        # Extract clickable elements
        clickable = []
        text_fields = []
        
        for window in ui_tree.get('windows', []):
            for elem in window.get('elements', []):
                if elem.get('role') in ['AXButton', 'AXLink']:
                    clickable.append(elem.get('title', ''))
                elif elem.get('role') in ['AXTextField', 'AXTextArea']:
                    text_fields.append(elem.get('value', ''))
        
        return {
            'app_name': app_info.get('name'),
            'app_bundle': app_info.get('bundle_id'),
            'focused_role': focused.get('role') if focused else None,
            'focused_value': focused.get('value') if focused else None,
            'clickable_elements': clickable,
            'text_fields': text_fields,
            'elements': [focused] if focused else []
        }
    
    def _process_saliency(self, saliency_metrics) -> Dict:
        """Process saliency metrics"""
        if not saliency_metrics:
            return {}
        
        return {
            'motion_score': saliency_metrics.motion_score,
            'ui_activity': saliency_metrics.ui_activity,
            'saliency_score': saliency_metrics.saliency_score,
            'should_process': saliency_metrics.should_process,
            'regions': saliency_metrics.regions_of_interest
        }
    
    def _generate_description(self, vlm_info: Dict, ocr_info: Dict, 
                             ui_info: Dict, saliency_info: Dict) -> str:
        """Generate unified description from all sources"""
        
        # If VLM has a good description, use it as primary
        if vlm_info.get('description'):
            vlm_desc = vlm_info['description']
            
            # Only add supplementary info if it's significant
            supplements = []
            
            # Add activity context from saliency if high motion
            if saliency_info.get('motion_score', 0) > 0.7:
                supplements.append("with active screen changes")
            elif saliency_info.get('ui_activity', 0) > 0.7:
                supplements.append("while interacting with UI")
            
            # Return VLM description, optionally with supplements
            if supplements:
                return f"{vlm_desc}, {', '.join(supplements)}"
            else:
                return vlm_desc
        
        # Fallback: construct description from other sources if no VLM
        descriptions = []
        
        # Start with application context
        if ui_info.get('app_name'):
            app_name = ui_info['app_name']
            descriptions.append(f"User is in {app_name}")
        
        # Add OCR context if significant text detected
        if ocr_info.get('high_conf_text'):
            text_preview = ' '.join(ocr_info['high_conf_text'][:3])
            if len(text_preview) > 50:
                text_preview = text_preview[:50] + "..."
            descriptions.append(f"viewing text: '{text_preview}'")
        
        # Add UI context
        if ui_info.get('focused_role'):
            role = ui_info['focused_role'].replace('AX', '')
            if ui_info.get('focused_value'):
                descriptions.append(f"focused on {role}: {ui_info['focused_value']}")
            else:
                descriptions.append(f"focused on {role}")
        
        # Add activity context from saliency
        if saliency_info.get('motion_score', 0) > 0.5:
            descriptions.append("with active screen changes")
        elif saliency_info.get('ui_activity', 0) > 0.5:
            descriptions.append("interacting with UI elements")
        
        # Combine descriptions
        if descriptions:
            return ' '.join(descriptions)
        else:
            return "Monitoring screen activity"
    
    def _calculate_confidence(self, vlm_info: Dict, ocr_info: Dict,
                             ui_info: Dict, saliency_info: Dict) -> float:
        """Calculate overall confidence from all sources"""
        confidences = []
        weights = []
        
        if vlm_info:
            confidences.append(vlm_info.get('confidence', 0.5))
            weights.append(self.source_weights['vlm'])
        
        if ocr_info.get('avg_confidence'):
            confidences.append(ocr_info['avg_confidence'])
            weights.append(self.source_weights['ocr'])
        
        if ui_info.get('app_name'):
            confidences.append(0.9)  # High confidence for UI data
            weights.append(self.source_weights['ui_tree'])
        
        if saliency_info.get('saliency_score'):
            confidences.append(saliency_info['saliency_score'])
            weights.append(self.source_weights['saliency'])
        
        if confidences:
            return np.average(confidences, weights=weights)
        return 0.5
    
    def _calculate_importance(self, vlm_info: Dict, ocr_info: Dict,
                            ui_info: Dict, saliency_info: Dict) -> float:
        """Calculate importance score"""
        importance = 0.0
        
        # High motion or UI activity
        if saliency_info.get('motion_score', 0) > 0.7:
            importance += 0.3
        if saliency_info.get('ui_activity', 0) > 0.7:
            importance += 0.2
        
        # Significant text detected
        if ocr_info.get('word_count', 0) > 20:
            importance += 0.2
        
        # User interaction detected
        if ui_info.get('focused_role') in ['AXTextField', 'AXButton']:
            importance += 0.2
        
        # Important applications
        important_apps = ['Terminal', 'VS Code', 'Xcode', 'Chrome', 'Safari']
        if ui_info.get('app_name') in important_apps:
            importance += 0.1
        
        return min(1.0, importance)
    
    def _detect_activity_type(self, description: str, ocr_info: Dict, ui_info: Dict) -> Optional[str]:
        """Detect the type of activity from VLM description primarily"""
        description_lower = description.lower()
        
        # Check OCR text
        ocr_text = ' '.join(ocr_info.get('text_lines', [])).lower()
        
        # Check against activity patterns
        scores = {}
        for activity, keywords in self.activity_patterns.items():
            score = sum(1 for keyword in keywords 
                       if keyword in description_lower or keyword in ocr_text)
            if score > 0:
                scores[activity] = score
        
        # Only use app-based detection if no VLM description available
        # Remove the automatic +3 bonus that was causing "browsing" spam
        if not description or description == "Monitoring screen activity":
            app_name = ui_info.get('app_name', '').lower()
            if 'code' in app_name or 'xcode' in app_name:
                scores['coding'] = scores.get('coding', 0) + 1
            elif 'chrome' in app_name or 'safari' in app_name:
                scores['browsing'] = scores.get('browsing', 0) + 1
        elif 'slack' in app_name or 'messages' in app_name:
            scores['messaging'] = scores.get('messaging', 0) + 3
        
        if scores:
            return max(scores.keys(), key=scores.get)
        return None
    
    def _extract_entities(self, vlm_info: Dict, ocr_info: Dict, ui_info: Dict) -> List[str]:
        """Extract key entities from all sources"""
        entities = set()
        
        # Generic terms to filter out
        generic_terms = {'User', 'The', 'Google', 'Chrome', 'Safari', 'Firefox', 
                        'Window', 'Screen', 'Desktop', 'App', 'Application', 
                        'QOO', 'UI', 'Element', 'Text', 'View'}
        
        # From VLM - extract meaningful entities
        vlm_entities = vlm_info.get('entities', [])
        for entity in vlm_entities:
            if entity not in generic_terms and len(entity) > 2:
                entities.add(entity)
        
        # From OCR - only high-confidence meaningful text
        for text in ocr_info.get('high_conf_text', [])[:3]:  # Limit to top 3
            words = text.split()
            for word in words:
                # Skip generic terms and short words
                if (word and word[0].isupper() and len(word) > 3 
                    and word not in generic_terms):
                    entities.add(word.strip('.,!?'))
        
        # Only add app name if it's specific (not a browser)
        app_name = ui_info.get('app_name', '')
        if app_name and app_name not in ['Google Chrome', 'Safari', 'Firefox', 'Edge']:
            entities.add(app_name)
        
        # Limit to most relevant, return as list
        return list(entities)[:5]  # Reduced from 10 to 5 for cleaner display
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Simple entity extraction from text"""
        entities = []
        
        # Generic terms to filter out
        generic_terms = {'User', 'The', 'Google', 'Chrome', 'Safari', 'Firefox',
                        'Window', 'Screen', 'Desktop', 'App', 'Application'}
        
        # Extract capitalized words (simple NER)
        words = text.split()
        for word in words:
            cleaned = word.strip('.,!?')
            if (cleaned and cleaned[0].isupper() and len(cleaned) > 3 
                and cleaned not in generic_terms):
                entities.append(cleaned)
        
        return entities[:5]
    
    def _suggest_actions(self, context: FusedContext) -> List[str]:
        """Generate suggested actions based on context"""
        suggestions = []
        
        # Based on activity type
        if context.activity_type == 'coding':
            if 'error' in context.description.lower():
                suggestions.append("Debug the error")
            suggestions.append("Save your work")
        
        elif context.activity_type == 'browsing':
            if len(context.ocr_text) > 50:
                suggestions.append("Bookmark this page")
            suggestions.append("Take notes")
        
        elif context.activity_type == 'writing':
            suggestions.append("Check grammar")
            suggestions.append("Save document")
        
        # Based on importance
        if context.importance > 0.7:
            suggestions.append("Pay attention to this activity")
        
        return suggestions[:3]
    
    def _apply_temporal_smoothing(self, context: FusedContext) -> FusedContext:
        """Apply temporal smoothing using history"""
        if len(self.context_history) < 3:
            return context
        
        # Get recent contexts
        recent = list(self.context_history)[-3:]
        
        # Smooth importance
        recent_importance = [c.importance for c in recent]
        context.importance = 0.7 * context.importance + 0.3 * np.mean(recent_importance)
        
        # Maintain activity type if consistent
        recent_activities = [c.activity_type for c in recent if c.activity_type]
        if recent_activities and all(a == recent_activities[0] for a in recent_activities):
            context.activity_type = recent_activities[0]
        
        return context
    
    def _detect_activity_simple(self, text: str) -> Optional[str]:
        """Simple activity detection from text"""
        text_lower = text.lower()
        
        for activity, keywords in self.activity_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return activity
        
        # Default based on common patterns
        if any(word in text_lower for word in ['editing', 'video', 'photo', 'image']):
            return 'editing'
        elif any(word in text_lower for word in ['terminal', 'command', 'shell']):
            return 'coding'
        elif any(word in text_lower for word in ['reading', 'viewing', 'watching']):
            return 'browsing'
            
        return None
    
    def fuse_simple(self, 
                   vlm_output: str,
                   frame_timestamp: Optional[float] = None) -> FusedContext:
        """
        Simple fusion using only VLM output.
        Optimized for memory and speed.
        
        Args:
            vlm_output: VLM description of the screen
            frame_timestamp: Timestamp of the frame
            
        Returns:
            FusedContext with VLM-based understanding
        """
        timestamp = frame_timestamp or time.time()
        
        # Extract key information from VLM output
        description = vlm_output
        
        # Try to extract application name from VLM output
        application = None
        if "in " in vlm_output.lower():
            # Simple extraction: "User is working in VS Code..."
            parts = vlm_output.lower().split("in ")
            if len(parts) > 1:
                app_part = parts[1].split()[0:2]  # Get first 2 words after "in"
                application = " ".join(app_part).strip(",. ")
        
        # Detect activity type from VLM output
        activity_type = self._detect_activity_simple(vlm_output)
        
        # Extract entities from VLM output  
        key_entities = self._extract_entities_from_text(vlm_output)
        
        # Create context
        context = FusedContext(
            timestamp=timestamp,
            description=description,
            confidence=0.8,  # Fixed confidence for VLM
            importance=0.7,  # Default importance
            vlm_output=vlm_output,
            application=application,
            activity_type=activity_type,
            key_entities=key_entities,
            sources_used=['vlm']
        )
        
        # Apply temporal smoothing
        context = self._apply_temporal_smoothing(context)
        
        # Add to history
        self.context_history.append(context)
        
        return context
    
    def get_recent_context(self, n: int = 5) -> List[FusedContext]:
        """Get recent context history"""
        return list(self.context_history)[-n:]
    
    def get_metrics(self) -> Dict:
        """Get fusion performance metrics"""
        metrics = {
            'history_size': len(self.context_history),
            'avg_fusion_time_ms': float(np.mean(self.fusion_times)) * 1000 if self.fusion_times else 0,
            'max_fusion_time_ms': float(np.max(self.fusion_times)) * 1000 if self.fusion_times else 0
        }
        
        if self.context_history:
            recent = list(self.context_history)[-10:]
            metrics['recent_avg_confidence'] = float(np.mean([c.confidence for c in recent]))
            metrics['recent_avg_importance'] = float(np.mean([c.importance for c in recent]))
            
            # Count activity types
            activity_counts = {}
            for c in recent:
                if c.activity_type:
                    activity_counts[c.activity_type] = activity_counts.get(c.activity_type, 0) + 1
            metrics['recent_activities'] = activity_counts
        
        return metrics


# Demo usage
if __name__ == "__main__":
    def demo():
        """Demonstrate context fusion"""
        print("🔀 Context Fusion Demo")
        print("=" * 40)
        
        # Create fusion system
        fusion = ContextFusion()
        
        # Simulate inputs
        vlm_output = "User is working in VS Code with multiple code files open"
        
        # Simulate OCR result
        class MockOCRResult:
            def __init__(self):
                self.regions = [
                    type('Region', (), {
                        'text': 'def process_frame(self, frame):',
                        'confidence': 0.95
                    })(),
                    type('Region', (), {
                        'text': 'import numpy as np',
                        'confidence': 0.88
                    })()
                ]
                self.total_words = 8
                self.average_confidence = 0.91
        
        ocr_result = MockOCRResult()
        
        # Simulate UI tree
        ui_tree = {
            'frontmost_app': {
                'name': 'Visual Studio Code',
                'bundle_id': 'com.microsoft.VSCode'
            },
            'focused_element': {
                'role': 'AXTextField',
                'value': 'process_frame'
            }
        }
        
        # Simulate saliency
        class MockSaliencyMetrics:
            def __init__(self):
                self.motion_score = 0.3
                self.ui_activity = 0.6
                self.saliency_score = 0.45
                self.should_process = True
                self.regions_of_interest = [(100, 200, 300, 400)]
        
        saliency_metrics = MockSaliencyMetrics()
        
        # Perform fusion
        print("\nPerforming context fusion...")
        context = fusion.fuse(
            vlm_output=vlm_output,
            ocr_result=ocr_result,
            ui_tree=ui_tree,
            saliency_metrics=saliency_metrics
        )
        
        # Display results
        print("\n📊 Fused Context:")
        print(f"  • Description: {context.description}")
        print(f"  • Confidence: {context.confidence:.2%}")
        print(f"  • Importance: {context.importance:.2%}")
        print(f"  • Application: {context.application}")
        print(f"  • Activity Type: {context.activity_type}")
        print(f"  • Key Entities: {context.key_entities}")
        print(f"  • Suggested Actions: {context.suggested_actions}")
        print(f"  • Sources Used: {context.sources_used}")
        print(f"  • Processing Time: {context.processing_time*1000:.1f}ms")
        
        # Show metrics
        print("\n⚡ Fusion Metrics:")
        for key, value in fusion.get_metrics().items():
            print(f"  • {key}: {value}")
    
    demo()