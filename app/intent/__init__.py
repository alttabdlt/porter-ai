"""
Intent module for Porter.AI.
Provides intent classification, policy-based routing, and action suggestions.
"""

from .intent_router import (
    IntentRouter,
    Intent,
    IntentType,
    RoutingPolicy,
    RoutingDecision,
    ActionSuggester,
    ActionSuggestion,
    IntentPattern
)

__all__ = [
    'IntentRouter',
    'Intent',
    'IntentType',
    'RoutingPolicy',
    'RoutingDecision',
    'ActionSuggester',
    'ActionSuggestion',
    'IntentPattern'
]