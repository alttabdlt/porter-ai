#!/usr/bin/env python3
"""
Intent Router for Porter.AI - Classifies and routes user intents.
Provides policy-based routing, confidence scoring, and action suggestions.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import deque, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Enumeration of possible intent types"""
    CODING = 'coding'
    BROWSING = 'browsing'
    COMMUNICATION = 'communication'
    PRODUCTIVITY = 'productivity'
    LEARNING = 'learning'
    ENTERTAINMENT = 'entertainment'
    DEBUGGING = 'debugging'
    MEETING = 'meeting'
    UNKNOWN = 'unknown'


@dataclass
class Intent:
    """Represents a classified intent with confidence"""
    primary_intent: str
    confidence: float
    secondary_intents: List[str] = field(default_factory=list)
    all_intents: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    memory_enhanced: bool = False
    age_minutes: float = 0
    
    def __post_init__(self):
        """Initialize all_intents if not provided"""
        if not self.all_intents:
            self.all_intents = {
                self.primary_intent: self.confidence
            }
            for intent in self.secondary_intents:
                if intent not in self.all_intents:
                    self.all_intents[intent] = self.confidence * 0.7


@dataclass
class RoutingPolicy:
    """Defines a routing policy for intent handling"""
    name: str
    condition: Callable[[Dict], bool]
    allowed_intents: List[str] = field(default_factory=list)
    blocked_intents: List[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class RoutingDecision:
    """Represents a routing decision based on policies"""
    action: str  # 'allow', 'block', 'redirect'
    reason: str
    policy: Optional[RoutingPolicy] = None
    blocked_intents: List[str] = field(default_factory=list)
    allowed_intents: List[str] = field(default_factory=list)


@dataclass
class ActionSuggestion:
    """Suggested action based on intent"""
    action: str
    description: str
    confidence: float
    priority: int = 0


@dataclass
class IntentPattern:
    """Detected pattern in intent history"""
    type: str
    intents: List[str]
    confidence: float
    description: str = ""


class IntentClassifier:
    """Classifies intents from descriptions using keyword and pattern matching"""
    
    def __init__(self):
        """Initialize intent classifier with keyword mappings"""
        self.intent_keywords = {
            IntentType.CODING.value: [
                'code', 'coding', 'programming', 'python', 'javascript', 'java',
                'function', 'class', 'variable', 'vs code', 'vscode', 'editor',
                'ide', 'syntax', 'compile', 'build', 'git', 'commit', 'pull request'
            ],
            IntentType.DEBUGGING.value: [
                'debug', 'debugging', 'error', 'exception', 'breakpoint', 'trace',
                'fix', 'bug', 'issue', 'problem', 'stack trace', 'console'
            ],
            IntentType.BROWSING.value: [
                'browse', 'browsing', 'web', 'website', 'google', 'search',
                'chrome', 'firefox', 'safari', 'browser', 'internet', 'reddit',
                'news', 'article'
            ],
            IntentType.LEARNING.value: [
                'documentation', 'docs', 'tutorial', 'learn', 'learning', 'guide',
                'stack overflow', 'mdn', 'reference', 'example', 'course', 'video'
            ],
            IntentType.COMMUNICATION.value: [
                'slack', 'email', 'message', 'chat', 'discord', 'teams', 'zoom',
                'call', 'meeting', 'conference', 'gmail', 'outlook', 'typing message'
            ],
            IntentType.PRODUCTIVITY.value: [
                'document', 'presentation', 'spreadsheet', 'excel', 'word',
                'powerpoint', 'google docs', 'notion', 'task', 'organize', 'edit'
            ],
            IntentType.ENTERTAINMENT.value: [
                'youtube', 'video', 'watch', 'watching', 'game', 'play', 'playing',
                'music', 'spotify', 'netflix', 'entertainment', 'reddit'
            ],
            IntentType.MEETING.value: [
                'zoom', 'teams', 'meet', 'video call', 'conference', 'participants',
                'screen share', 'presentation', 'meeting'
            ]
        }
        
        # Compile regex patterns for efficiency
        self.patterns = {}
        for intent, keywords in self.intent_keywords.items():
            pattern = '|'.join(re.escape(kw) for kw in keywords)
            self.patterns[intent] = re.compile(pattern, re.IGNORECASE)
    
    def classify(self, description: str) -> Intent:
        """
        Classify intent from description text.
        
        Args:
            description: Text description to classify
            
        Returns:
            Intent object with classification results
        """
        if not description:
            return Intent(
                primary_intent=IntentType.UNKNOWN.value,
                confidence=0.0,
                description=description
            )
        
        # Score each intent type
        intent_scores = {}
        description_lower = description.lower()
        
        for intent_type, pattern in self.patterns.items():
            matches = pattern.findall(description_lower)
            if matches:
                # Score based on number and uniqueness of matches
                unique_matches = set(matches)
                score = len(matches) * 0.1 + len(unique_matches) * 0.2
                intent_scores[intent_type] = min(score, 1.0)
        
        if not intent_scores:
            # No clear intent found
            return Intent(
                primary_intent=IntentType.UNKNOWN.value,
                confidence=0.3,
                description=description
            )
        
        # Sort intents by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Primary intent is the highest scoring
        primary_intent = sorted_intents[0][0]
        primary_confidence = sorted_intents[0][1]
        
        # Boost confidence for very clear descriptions
        if len(sorted_intents) == 1 and primary_confidence > 0.5:
            primary_confidence = min(primary_confidence * 1.3, 0.95)
        elif len(sorted_intents) > 1 and primary_confidence > sorted_intents[1][1] * 2:
            primary_confidence = min(primary_confidence * 1.2, 0.9)
        
        # Secondary intents are others with significant scores
        secondary_intents = [
            intent for intent, score in sorted_intents[1:]
            if score > primary_confidence * 0.5
        ]
        
        return Intent(
            primary_intent=primary_intent,
            confidence=primary_confidence,
            secondary_intents=secondary_intents,
            all_intents=dict(sorted_intents),
            description=description
        )


class IntentRouter:
    """
    Main intent router that classifies intents and makes routing decisions.
    """
    
    def __init__(
        self,
        policies: Optional[List[RoutingPolicy]] = None,
        memory_layer: Optional[Any] = None,
        history_size: int = 100
    ):
        """
        Initialize intent router.
        
        Args:
            policies: List of routing policies
            memory_layer: Optional memory layer for context
            history_size: Size of intent history to maintain
        """
        self.classifier = IntentClassifier()
        self.policies = sorted(policies or [], key=lambda p: p.priority, reverse=True)
        self.memory_layer = memory_layer
        self.history = deque(maxlen=history_size)
        self.history_size = history_size
    
    async def classify_intent(self, description: str) -> Intent:
        """
        Classify intent from description.
        
        Args:
            description: Text description to classify
            
        Returns:
            Classified intent
        """
        return self.classifier.classify(description)
    
    async def classify_with_memory(self, description: str) -> Intent:
        """
        Classify intent with memory context enhancement.
        
        Args:
            description: Current description
            
        Returns:
            Memory-enhanced intent
        """
        # Get base classification
        intent = await self.classify_intent(description)
        
        if self.memory_layer:
            # Search for similar past contexts
            similar = await self.memory_layer.search_similar(description, n_results=3)
            
            if similar:
                # Use memory to boost confidence if consistent
                memory_intents = []
                for context in similar:
                    if 'activity' in context:
                        activity = context['activity']
                        if activity in ['coding', 'debugging', 'testing']:
                            memory_intents.append('coding')
                        elif activity in ['browsing', 'research']:
                            memory_intents.append('browsing')
                        elif activity in ['messaging', 'meeting']:
                            memory_intents.append('communication')
                
                if memory_intents:
                    # If memory agrees with classification, boost confidence
                    most_common = Counter(memory_intents).most_common(1)[0][0]
                    if intent.primary_intent == most_common:
                        intent.confidence = min(intent.confidence * 1.3, 0.95)
                    elif intent.primary_intent == IntentType.UNKNOWN.value:
                        # Memory can help resolve unknown intents
                        intent.primary_intent = most_common
                        intent.confidence = 0.7
                    
                    intent.memory_enhanced = True
        
        return intent
    
    async def route_intent(
        self,
        intent: Intent,
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """
        Make routing decision based on intent and policies.
        
        Args:
            intent: Classified intent
            context: Current context
            
        Returns:
            Routing decision
        """
        # Check each policy in priority order
        for policy in self.policies:
            if policy.condition(context):
                # Policy applies, check if intent is allowed or blocked
                if policy.blocked_intents and intent.primary_intent in policy.blocked_intents:
                    return RoutingDecision(
                        action='block',
                        reason=policy.name,
                        policy=policy,
                        blocked_intents=policy.blocked_intents
                    )
                elif policy.allowed_intents:
                    if intent.primary_intent in policy.allowed_intents:
                        return RoutingDecision(
                            action='allow',
                            reason=policy.name,
                            policy=policy,
                            allowed_intents=policy.allowed_intents
                        )
                    else:
                        # Intent not in allowed list
                        return RoutingDecision(
                            action='block',
                            reason=policy.name,
                            policy=policy,
                            blocked_intents=[intent.primary_intent]
                        )
        
        # No policy matched, default allow
        return RoutingDecision(
            action='allow',
            reason='default'
        )
    
    def get_decayed_confidence(self, intent: Intent) -> float:
        """
        Get confidence with time decay applied.
        
        Args:
            intent: Intent to decay
            
        Returns:
            Decayed confidence value
        """
        if not hasattr(intent, 'age_minutes'):
            # Calculate age if not set
            age = (datetime.now() - intent.timestamp).total_seconds() / 60
        else:
            age = intent.age_minutes
        
        # Exponential decay: confidence * e^(-decay_rate * time)
        decay_rate = 0.1  # Confidence halves every ~7 minutes
        import math
        decayed = intent.confidence * math.exp(-decay_rate * age)
        
        return max(decayed, 0.1)  # Minimum confidence of 0.1
    
    async def record_intent(self, intent: Intent):
        """
        Record intent in history.
        
        Args:
            intent: Intent to record
        """
        self.history.append(intent)
    
    def get_intent_history(self) -> List[Intent]:
        """
        Get intent history.
        
        Returns:
            List of recent intents
        """
        return list(self.history)
    
    async def detect_pattern(self) -> Optional[IntentPattern]:
        """
        Detect patterns in intent history.
        
        Returns:
            Detected pattern or None
        """
        if len(self.history) < 3:
            return None
        
        recent = list(self.history)[-10:]
        intent_sequence = [i.primary_intent for i in recent]
        
        # Check for development cycle pattern
        if 'coding' in intent_sequence and 'debugging' in intent_sequence:
            if intent_sequence.count('coding') >= 2:
                return IntentPattern(
                    type='development_cycle',
                    intents=['coding', 'debugging', 'testing'],
                    confidence=0.8,
                    description='User is in active development cycle'
                )
        
        # Check for research pattern
        if 'browsing' in intent_sequence and 'learning' in intent_sequence:
            return IntentPattern(
                type='research',
                intents=['browsing', 'learning'],
                confidence=0.7,
                description='User is researching or learning'
            )
        
        # Check for communication pattern
        if intent_sequence.count('communication') >= 3:
            return IntentPattern(
                type='communication_focus',
                intents=['communication'],
                confidence=0.8,
                description='User is focused on communication'
            )
        
        return None
    
    def analyze_transition(
        self,
        from_intent: Intent,
        to_intent: Intent
    ) -> Dict[str, Any]:
        """
        Analyze transition between intents.
        
        Args:
            from_intent: Starting intent
            to_intent: Ending intent
            
        Returns:
            Transition analysis
        """
        transition_type = 'unknown'
        likelihood = 0.5
        
        # Common transitions
        common_transitions = {
            ('coding', 'debugging'): ('troubleshooting', 0.8),
            ('coding', 'browsing'): ('research', 0.7),
            ('communication', 'meeting'): ('scheduled', 0.9),
            ('browsing', 'coding'): ('implementation', 0.6),
            ('productivity', 'communication'): ('collaboration', 0.7)
        }
        
        key = (from_intent.primary_intent, to_intent.primary_intent)
        if key in common_transitions:
            transition_type, likelihood = common_transitions[key]
        
        return {
            'type': transition_type,
            'likelihood': likelihood,
            'from': from_intent.primary_intent,
            'to': to_intent.primary_intent
        }
    
    async def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary of intent history.
        
        Returns:
            Summary statistics
        """
        if not self.history:
            return {
                'total_intents': 0,
                'dominant_intent': None,
                'intent_distribution': {}
            }
        
        # Count intent frequencies
        intent_counts = Counter(i.primary_intent for i in self.history)
        total = len(self.history)
        
        # Calculate distribution
        distribution = {
            intent: count / total
            for intent, count in intent_counts.items()
        }
        
        return {
            'total_intents': total,
            'dominant_intent': intent_counts.most_common(1)[0][0],
            'intent_distribution': distribution,
            'unique_intents': len(intent_counts)
        }
    
    async def persist_intent(self, intent: Intent):
        """
        Persist intent to memory layer.
        
        Args:
            intent: Intent to persist
        """
        if self.memory_layer:
            context = {
                'timestamp': intent.timestamp.isoformat(),
                'intent': intent.primary_intent,
                'confidence': intent.confidence,
                'description': intent.description,
                'secondary_intents': intent.secondary_intents,
                'importance': intent.confidence
            }
            
            await self.memory_layer.store_context(context)


class ActionSuggester:
    """
    Suggests actions based on classified intents.
    """
    
    def __init__(self):
        """Initialize action suggester with predefined suggestions"""
        self.intent_actions = {
            'coding': [
                ActionSuggestion('enable_focus_mode', 'Enable focus mode to minimize distractions', 0.8, 10),
                ActionSuggestion('mute_notifications', 'Mute notifications while coding', 0.7, 8),
                ActionSuggestion('start_pomodoro', 'Start Pomodoro timer for focused work', 0.6, 5)
            ],
            'communication': [
                ActionSuggestion('close_distracting_tabs', 'Close distracting browser tabs', 0.7, 8),
                ActionSuggestion('enable_do_not_disturb', 'Enable do not disturb mode', 0.8, 10),
                ActionSuggestion('prepare_meeting_notes', 'Prepare meeting notes document', 0.6, 5)
            ],
            'browsing': [
                ActionSuggestion('set_timer', 'Set timer to limit browsing time', 0.6, 5),
                ActionSuggestion('bookmark_useful', 'Bookmark useful resources', 0.5, 3)
            ],
            'entertainment': [
                ActionSuggestion('reminder_return_to_work', 'Set reminder to return to work', 0.7, 8),
                ActionSuggestion('track_break_time', 'Track break time duration', 0.6, 5)
            ]
        }
    
    async def suggest_actions(
        self,
        intent: Intent,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ActionSuggestion]:
        """
        Suggest actions based on intent and context.
        
        Args:
            intent: Classified intent
            context: Optional context information
            
        Returns:
            List of suggested actions
        """
        suggestions = []
        
        # Get base suggestions for intent type
        if intent.primary_intent in self.intent_actions:
            suggestions.extend(self.intent_actions[intent.primary_intent])
        
        # Context-aware modifications
        if context:
            # During work hours, suggest more productivity actions
            if 'hour' in context and 9 <= context['hour'] <= 17:
                if intent.primary_intent in ['browsing', 'entertainment']:
                    suggestions.append(
                        ActionSuggestion(
                            'reminder_return_to_work',
                            'Consider returning to productive tasks',
                            0.8, 10
                        )
                    )
            
            # Long duration detection
            if 'duration_minutes' in context and context['duration_minutes'] > 30:
                if intent.primary_intent == 'entertainment':
                    suggestions.append(
                        ActionSuggestion(
                            'reminder_return_to_work',
                            'You\'ve been on break for a while',
                            0.9, 15
                        )
                    )
        
        # Sort by priority
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        
        return suggestions