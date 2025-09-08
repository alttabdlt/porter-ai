#!/usr/bin/env python3
"""
Memory Layer for Porter.AI - Persistent, queryable context storage.
Uses ChromaDB for vector storage and semantic search.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class MemoryLayer:
    """
    Memory layer for storing and querying context history.
    Provides persistent storage, semantic search, and memory-based enhancement.
    """
    
    def __init__(
        self,
        persist_directory: str = "./memory_store",
        collection_name: str = "porter_contexts",
        max_contexts: int = 10000
    ):
        """
        Initialize memory layer with ChromaDB.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            max_contexts: Maximum number of contexts to store
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.max_contexts = max_contexts
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection with embedding function
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using the configured embedding function.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = self.embedding_function([text])
        if embeddings and len(embeddings) > 0:
            # Convert to list if it's a numpy array
            if hasattr(embeddings[0], 'tolist'):
                return embeddings[0].tolist()
            return list(embeddings[0])
        return []
    
    async def store_context(
        self,
        context: Dict[str, Any],
        screenshot: Optional[np.ndarray] = None
    ) -> str:
        """
        Store a context in memory with optional screenshot.
        
        Args:
            context: Context dictionary with timestamp, description, etc.
            screenshot: Optional screenshot array
            
        Returns:
            Context ID
        """
        # Generate unique ID
        context_id = str(uuid.uuid4())
        
        # Prepare document text for embedding
        document = context.get('description', '')
        if 'activity' in context:
            document += f" Activity: {context['activity']}"
        if 'application' in context:
            document += f" Application: {context['application']}"
        
        # Prepare metadata
        metadata = {
            'timestamp': context.get('timestamp', datetime.now().isoformat()),
            'activity': context.get('activity', 'unknown'),
            'application': context.get('application', 'unknown'),
            'importance': float(context.get('importance', 0.5))
        }
        
        # Add screenshot metadata if provided
        if screenshot is not None:
            metadata['has_screenshot'] = True
            metadata['screenshot_shape'] = json.dumps(list(screenshot.shape))
            # Store screenshot shape in context for retrieval
            context['screenshot_shape'] = list(screenshot.shape)
        
        # Add additional context fields to metadata
        for key, value in context.items():
            if key not in metadata and key != 'description':
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = json.dumps(value)
        
        # Store in ChromaDB
        self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[context_id]
        )
        
        # Check if we need to cleanup old contexts
        await self._cleanup_if_needed()
        
        logger.debug(f"Stored context {context_id}: {document[:50]}...")
        return context_id
    
    async def get_context(self, context_id: str) -> Optional[Dict]:
        """
        Retrieve a specific context by ID.
        
        Args:
            context_id: Context ID to retrieve
            
        Returns:
            Context dictionary or None if not found
        """
        try:
            result = self.collection.get(ids=[context_id])
            
            if result['ids']:
                metadata = result['metadatas'][0]
                context = {
                    'id': context_id,
                    'description': result['documents'][0],
                    **metadata
                }
                
                # Parse JSON fields
                if 'screenshot_shape' in context:
                    context['screenshot_shape'] = json.loads(context['screenshot_shape'])
                
                return context
            
        except Exception as e:
            logger.error(f"Error retrieving context {context_id}: {e}")
        
        return None
    
    async def search_similar(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search for similar contexts using semantic search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar contexts
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            contexts = []
            for i in range(len(results['ids'][0])):
                context = {
                    'id': results['ids'][0][i],
                    'description': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0,
                    **results['metadatas'][0][i]
                }
                contexts.append(context)
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error searching similar contexts: {e}")
            return []
    
    async def query_time_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """
        Query contexts within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of contexts in time range
        """
        try:
            # Get all contexts (limited for performance)
            results = self.collection.get(
                limit=1000,
                where={
                    "$and": [
                        {"timestamp": {"$gte": start_time.isoformat()}},
                        {"timestamp": {"$lte": end_time.isoformat()}}
                    ]
                }
            )
            
            contexts = []
            for i in range(len(results['ids'])):
                context = {
                    'id': results['ids'][i],
                    'description': results['documents'][i],
                    **results['metadatas'][i]
                }
                contexts.append(context)
            
            # Sort by timestamp
            contexts.sort(key=lambda x: x.get('timestamp', ''))
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error querying time range: {e}")
            return []
    
    async def query_by_application(self, application: str) -> List[Dict]:
        """
        Query contexts by application.
        
        Args:
            application: Application name
            
        Returns:
            List of contexts from the application
        """
        try:
            results = self.collection.get(
                where={"application": application},
                limit=100
            )
            
            contexts = []
            for i in range(len(results['ids'])):
                context = {
                    'id': results['ids'][i],
                    'description': results['documents'][i],
                    **results['metadatas'][i]
                }
                contexts.append(context)
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error querying by application: {e}")
            return []
    
    async def query_important(self, threshold: float = 0.8) -> List[Dict]:
        """
        Query high-importance contexts.
        
        Args:
            threshold: Importance threshold (0-1)
            
        Returns:
            List of important contexts
        """
        try:
            results = self.collection.get(
                where={"importance": {"$gte": threshold}},
                limit=100
            )
            
            contexts = []
            for i in range(len(results['ids'])):
                context = {
                    'id': results['ids'][i],
                    'description': results['documents'][i],
                    **results['metadatas'][i]
                }
                contexts.append(context)
            
            # Sort by importance
            contexts.sort(key=lambda x: x.get('importance', 0), reverse=True)
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error querying important contexts: {e}")
            return []
    
    async def get_activity_summary(self, hours: int = 24) -> Dict:
        """
        Generate activity summary for the last N hours.
        
        Args:
            hours: Number of hours to summarize
            
        Returns:
            Summary dictionary with statistics
        """
        start_time = datetime.now() - timedelta(hours=hours)
        end_time = datetime.now()
        
        contexts = await self.query_time_range(start_time, end_time)
        
        if not contexts:
            return {
                'total_contexts': 0,
                'activities': {},
                'applications': {},
                'average_importance': 0
            }
        
        # Calculate statistics
        activities = {}
        applications = {}
        total_importance = 0
        
        for context in contexts:
            # Count activities
            activity = context.get('activity', 'unknown')
            activities[activity] = activities.get(activity, 0) + 1
            
            # Count applications
            app = context.get('application', 'unknown')
            applications[app] = applications.get(app, 0) + 1
            
            # Sum importance
            total_importance += context.get('importance', 0)
        
        return {
            'total_contexts': len(contexts),
            'activities': activities,
            'applications': applications,
            'average_importance': total_importance / len(contexts) if contexts else 0,
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            }
        }
    
    async def enhance_with_memory(
        self,
        current_description: str,
        lookback_minutes: int = 30
    ) -> Dict:
        """
        Enhance current context with recent memory.
        
        Args:
            current_description: Current context description
            lookback_minutes: How far back to look in memory
            
        Returns:
            Enhanced context with memory suggestions
        """
        # Get recent contexts
        start_time = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_contexts = await self.query_time_range(
            start_time,
            datetime.now()
        )
        
        # Search for similar contexts
        similar_contexts = await self.search_similar(
            current_description,
            n_results=3
        )
        
        # Build suggested context
        suggested_context = current_description
        
        if recent_contexts:
            # Add recent activity context
            recent_activities = [c.get('activity', '') for c in recent_contexts[-3:]]
            recent_apps = [c.get('application', '') for c in recent_contexts[-3:]]
            
            if recent_activities:
                suggested_context += f" Recent activities: {', '.join(set(recent_activities))}"
            if recent_apps:
                suggested_context += f" Recent applications: {', '.join(set(recent_apps))}"
        
        return {
            'current': current_description,
            'suggested_context': suggested_context,
            'recent_contexts': recent_contexts[-5:] if recent_contexts else [],
            'similar_contexts': similar_contexts
        }
    
    async def get_context_count(self) -> int:
        """
        Get total number of stored contexts.
        
        Returns:
            Number of contexts
        """
        try:
            # ChromaDB doesn't have a direct count method, so we use a query
            result = self.collection.get(limit=1)
            # This is an approximation - for exact count would need to query all
            return len(result['ids'])
        except:
            return 0
    
    async def _cleanup_if_needed(self):
        """
        Clean up old contexts if over the limit.
        Keeps the most recent contexts.
        """
        try:
            # Get all contexts to check count
            results = self.collection.get(limit=self.max_contexts + 100)
            
            if len(results['ids']) > self.max_contexts:
                # Sort by timestamp and remove oldest
                contexts = []
                for i in range(len(results['ids'])):
                    contexts.append({
                        'id': results['ids'][i],
                        'timestamp': results['metadatas'][i].get('timestamp', '')
                    })
                
                # Sort by timestamp
                contexts.sort(key=lambda x: x['timestamp'])
                
                # Delete oldest contexts
                to_delete = contexts[:len(contexts) - self.max_contexts]
                if to_delete:
                    self.collection.delete(ids=[c['id'] for c in to_delete])
                    logger.info(f"Cleaned up {len(to_delete)} old contexts")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class MemoryIntegration:
    """
    Integration helper for connecting memory layer with streaming pipeline.
    """
    
    def __init__(self, memory_layer: MemoryLayer):
        """
        Initialize memory integration.
        
        Args:
            memory_layer: MemoryLayer instance
        """
        self.memory = memory_layer
        self.last_store_time = datetime.now()
        self.store_interval = timedelta(seconds=5)  # Store every 5 seconds
    
    async def process_context(
        self,
        context: Dict,
        force_store: bool = False
    ) -> Optional[str]:
        """
        Process and potentially store a context from the pipeline.
        
        Args:
            context: Context from the streaming pipeline
            force_store: Force storage regardless of interval
            
        Returns:
            Context ID if stored, None otherwise
        """
        now = datetime.now()
        
        # Check if we should store (based on interval or importance)
        should_store = force_store or \
                      (now - self.last_store_time) >= self.store_interval or \
                      context.get('importance', 0) >= 0.8
        
        if should_store:
            # Prepare context for storage
            storage_context = {
                'timestamp': context.get('timestamp', now.isoformat()),
                'description': context.get('vlm_output', ''),
                'importance': context.get('importance', 0.5),
                'activity': self._infer_activity(context),
                'application': self._infer_application(context)
            }
            
            # Store context
            context_id = await self.memory.store_context(storage_context)
            self.last_store_time = now
            
            return context_id
        
        return None
    
    def _infer_activity(self, context: Dict) -> str:
        """Infer activity type from context."""
        description = context.get('vlm_output', '').lower()
        
        if 'code' in description or 'programming' in description:
            return 'coding'
        elif 'youtube' in description or 'video' in description:
            return 'watching'
        elif 'slack' in description or 'message' in description:
            return 'messaging'
        elif 'browser' in description or 'chrome' in description:
            return 'browsing'
        elif 'photoshop' in description or 'design' in description:
            return 'design'
        else:
            return 'working'
    
    def _infer_application(self, context: Dict) -> str:
        """Infer application from context."""
        description = context.get('vlm_output', '').lower()
        
        apps = {
            'vs code': ['vs code', 'visual studio code', 'vscode'],
            'Chrome': ['chrome', 'browser'],
            'Slack': ['slack'],
            'Terminal': ['terminal', 'console'],
            'Photoshop': ['photoshop'],
            'YouTube': ['youtube']
        }
        
        for app_name, keywords in apps.items():
            if any(keyword in description for keyword in keywords):
                return app_name
        
        return 'Unknown'