#!/usr/bin/env python3
"""
Enhanced Memory Layer with better embeddings and smarter queries.
Uses sentence-transformers for improved semantic search.
"""

import logging
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from .memory_layer import MemoryLayer

logger = logging.getLogger(__name__)


class EnhancedMemoryLayer(MemoryLayer):
    """
    Enhanced memory layer with sentence-transformers and richer metadata.
    Provides better semantic search and temporal weighting.
    """
    
    def __init__(
        self,
        persist_directory: str = "./memory_store",
        collection_name: str = "porter_contexts",
        max_contexts: int = 10000,
        embedding_model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize enhanced memory layer with sentence-transformers.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            max_contexts: Maximum number of contexts to store
            embedding_model_name: Sentence-transformer model to use
        """
        # Initialize base class (but we'll override the embedding function)
        super().__init__(persist_directory, collection_name, max_contexts)
        
        # Initialize sentence-transformer model (use CPU to avoid MPS issues)
        self.embedding_model_name = embedding_model_name
        import torch
        self.sentence_model = SentenceTransformer(
            embedding_model_name,
            device='cpu'  # Force CPU to avoid Metal/MPS issues on macOS
        )
        
        # Override the collection with our custom embedding function
        self._reinitialize_collection()
        
        logger.info(f"Enhanced memory layer initialized with {embedding_model_name}")
    
    def _reinitialize_collection(self):
        """Reinitialize collection with sentence-transformer embeddings"""
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        
        # Create new collection with sentence-transformer embedding function
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=None,  # We'll provide embeddings directly
            metadata={"hnsw:space": "cosine"}
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using sentence-transformers.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text:
            # Return zero vector for empty text
            return [0.0] * 384  # all-MiniLM-L6-v2 dimension
        
        # Generate embedding using sentence-transformer
        embedding = self.sentence_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    async def store_enhanced_context(
        self,
        context: Dict[str, Any],
        screenshot: Optional[np.ndarray] = None
    ) -> str:
        """
        Store context with enhanced metadata extraction.
        
        Args:
            context: Context dictionary with enhanced fields
            screenshot: Optional screenshot array
            
        Returns:
            Context ID
        """
        # Extract enhanced metadata
        enhanced_context = context.copy()
        
        # Extract file name from window title if present
        if 'window_title' in context:
            file_name = self._extract_file_name(context['window_title'])
            if file_name:
                enhanced_context['file_name'] = file_name
        
        # Extract domain from URL if present
        if 'url' in context:
            domain = self._extract_domain(context['url'])
            if domain:
                enhanced_context['domain'] = domain
        
        # Extract project from window title or path
        if 'window_title' in context:
            project = self._extract_project(context['window_title'])
            if project:
                enhanced_context['project'] = project
        
        # Store using base method with enhanced context
        return await self.store_context(enhanced_context, screenshot)
    
    async def store_context(
        self,
        context: Dict[str, Any],
        screenshot: Optional[np.ndarray] = None
    ) -> str:
        """
        Override to use sentence-transformer embeddings directly.
        """
        # Generate unique ID
        import uuid
        context_id = str(uuid.uuid4())
        
        # Prepare document text for embedding
        document = context.get('description', '')
        if 'activity' in context:
            document += f" Activity: {context['activity']}"
        if 'application' in context:
            document += f" Application: {context['application']}"
        if 'window_title' in context:
            document += f" Window: {context['window_title']}"
        
        # Generate embedding using sentence-transformer
        embedding = self.generate_embedding(document)
        
        # Prepare metadata
        metadata = {
            'timestamp': context.get('timestamp', datetime.now().isoformat()),
            'activity': context.get('activity', 'unknown'),
            'application': context.get('application', 'unknown'),
            'importance': float(context.get('importance', 0.5))
        }
        
        # Add all other context fields to metadata
        for key, value in context.items():
            if key not in metadata and key != 'description':
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = json.dumps(value)
        
        # Store in ChromaDB with explicit embedding
        self.collection.add(
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[context_id]
        )
        
        # Check if we need to cleanup old contexts
        await self._cleanup_if_needed()
        
        logger.debug(f"Stored enhanced context {context_id}")
        return context_id
    
    async def search_similar_weighted(
        self,
        query: str,
        n_results: int = 5,
        use_temporal_weight: bool = True,
        recency_weight: float = 0.3
    ) -> List[Dict]:
        """
        Search with temporal weighting for recency.
        
        Args:
            query: Search query
            n_results: Number of results
            use_temporal_weight: Whether to apply temporal weighting
            recency_weight: Weight for recency (0-1)
            
        Returns:
            List of contexts with weighted scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search with more results to apply weighting
        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2  # Get more for filtering
        )
        
        if not raw_results['ids'] or not raw_results['ids'][0]:
            return []
        
        contexts = []
        now = datetime.now()
        
        for i in range(len(raw_results['ids'][0])):
            context = {
                'id': raw_results['ids'][0][i],
                'description': raw_results['documents'][0][i],
                'distance': raw_results['distances'][0][i] if 'distances' in raw_results else 0,
                **raw_results['metadatas'][0][i]
            }
            
            # Calculate similarity score (1 - distance for cosine)
            similarity_score = 1 - context['distance']
            
            if use_temporal_weight:
                # Calculate temporal decay
                timestamp = datetime.fromisoformat(context.get('timestamp', now.isoformat()))
                age_hours = (now - timestamp).total_seconds() / 3600
                
                # Exponential decay: e^(-decay_rate * age)
                decay_rate = 0.05  # Adjust for desired decay speed
                temporal_score = math.exp(-decay_rate * age_hours)
                
                # Combine scores
                weighted_score = (
                    (1 - recency_weight) * similarity_score +
                    recency_weight * temporal_score
                )
            else:
                weighted_score = similarity_score
            
            context['weighted_score'] = weighted_score
            contexts.append(context)
        
        # Sort by weighted score and return top n
        contexts.sort(key=lambda x: x['weighted_score'], reverse=True)
        return contexts[:n_results]
    
    async def query_by_project(self, project: str) -> List[Dict]:
        """
        Query contexts by project name.
        
        Args:
            project: Project name
            
        Returns:
            List of contexts from the project
        """
        try:
            results = self.collection.get(
                where={"project": project},
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
            
            # Sort by timestamp
            contexts.sort(key=lambda x: x.get('timestamp', ''))
            return contexts
            
        except Exception as e:
            logger.error(f"Error querying by project: {e}")
            return []
    
    async def get_activity_clusters(self) -> Dict[str, List[Dict]]:
        """
        Get contexts clustered by activity type.
        
        Returns:
            Dictionary of activity clusters
        """
        # Get all contexts (limited for performance)
        results = self.collection.get(limit=1000)
        
        clusters = defaultdict(list)
        
        for i in range(len(results['ids'])):
            context = {
                'id': results['ids'][i],
                'description': results['documents'][i],
                **results['metadatas'][i]
            }
            
            activity = context.get('activity', 'unknown')
            clusters[activity].append(context)
        
        return dict(clusters)
    
    async def extract_workflow_patterns(
        self,
        min_occurrences: int = 2,
        window_size: int = 6
    ) -> List[Dict]:
        """
        Extract repeated workflow patterns from context history.
        
        Args:
            min_occurrences: Minimum times pattern must occur
            window_size: Size of pattern window
            
        Returns:
            List of detected patterns
        """
        # Get recent contexts
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        contexts = await self.query_time_range(start_time, end_time)
        
        if len(contexts) < window_size:
            return []
        
        # Extract activity sequences
        activities = [c.get('activity', 'unknown') for c in contexts]
        
        # Find patterns using sliding window
        patterns = Counter()
        for i in range(len(activities) - window_size + 1):
            pattern = tuple(activities[i:i + window_size])
            patterns[pattern] += 1
        
        # Filter by minimum occurrences
        frequent_patterns = []
        for pattern, count in patterns.items():
            if count >= min_occurrences:
                # Extract unique activities in pattern
                unique_activities = list(set(pattern))
                frequent_patterns.append({
                    'activities': unique_activities,
                    'sequence': list(pattern),
                    'occurrences': count,
                    'confidence': count / len(activities)
                })
        
        # Sort by occurrences
        frequent_patterns.sort(key=lambda x: x['occurrences'], reverse=True)
        return frequent_patterns
    
    async def get_semantic_activity_groups(self) -> Dict[str, List[Dict]]:
        """
        Group activities by semantic similarity.
        
        Returns:
            Dictionary of semantic groups
        """
        # Get recent contexts
        results = self.collection.get(limit=500)
        
        if not results['ids']:
            return {}
        
        # Define semantic groups based on keywords
        semantic_groups = {
            'development': ['code', 'coding', 'debug', 'programming', 'compile', 'build'],
            'learning': ['documentation', 'tutorial', 'learn', 'reference', 'guide'],
            'communication': ['email', 'slack', 'message', 'chat', 'meeting', 'call']
        }
        
        grouped = defaultdict(list)
        
        for i in range(len(results['ids'])):
            description = results['documents'][i].lower()
            context = {
                'id': results['ids'][i],
                'description': results['documents'][i],
                **results['metadatas'][i]
            }
            
            # Assign to semantic group
            assigned = False
            for group_name, keywords in semantic_groups.items():
                if any(keyword in description for keyword in keywords):
                    grouped[group_name].append(context)
                    assigned = True
                    break
            
            if not assigned:
                grouped['other'].append(context)
        
        return dict(grouped)
    
    async def store_batch_contexts(
        self,
        contexts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store multiple contexts efficiently in batch.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            List of context IDs
        """
        if not contexts:
            return []
        
        import uuid
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for context in contexts:
            # Generate ID
            context_id = str(uuid.uuid4())
            ids.append(context_id)
            
            # Prepare document
            document = context.get('description', '')
            if 'window_title' in context:
                document += f" Window: {context['window_title']}"
            documents.append(document)
            
            # Generate embedding
            embedding = self.generate_embedding(document)
            embeddings.append(embedding)
            
            # Prepare metadata
            metadata = {
                'timestamp': context.get('timestamp', datetime.now().isoformat()),
                'importance': float(context.get('importance', 0.5))
            }
            
            for key, value in context.items():
                if key not in metadata and key != 'description':
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = json.dumps(value)
            
            metadatas.append(metadata)
        
        # Batch add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.debug(f"Batch stored {len(ids)} contexts")
        return ids
    
    def _extract_file_name(self, window_title: str) -> Optional[str]:
        """Extract file name from window title."""
        # Common patterns: "file.py - Editor", "file.js | Project"
        import re
        patterns = [
            r'^([^-|]+\.\w+)',  # file.ext at start
            r'([^/\\]+\.\w+)\s*[-|]',  # file.ext before separator
        ]
        
        for pattern in patterns:
            match = re.search(pattern, window_title)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return None
    
    def _extract_project(self, window_title: str) -> Optional[str]:
        """Extract project name from window title."""
        # Common patterns: "file - project - Editor", "project/file"
        import re
        
        # Look for common project indicators
        if ' - ' in window_title:
            parts = window_title.split(' - ')
            if len(parts) >= 2:
                # Often project is the second part
                project_candidate = parts[-2].strip()
                # Filter out common app names
                if project_candidate not in ['Visual Studio Code', 'Chrome', 'Terminal']:
                    return project_candidate
        
        # Look for path-like structures
        if '/' in window_title:
            parts = window_title.split('/')
            # Project might be in path
            for part in parts:
                if part and not part.startswith('.'):
                    return part.strip()
        
        return None