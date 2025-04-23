#!/usr/bin/env python

"""
Embedding Engine for Containerized Agents

Provides embedding generation for agents in containerized environments
without requiring the full semsubscription package.
"""

import logging
import os
import numpy as np
from typing import List, Union, Optional

# Conditionally import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Handles embedding generation for containerized agents
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize the embedding engine
        
        Args:
            model_name: Name of the pre-trained sentence transformer model or path to model directory
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the embedding model
        """
        if not HAVE_SENTENCE_TRANSFORMERS:
            logger.error("sentence-transformers package not installed. Please install it to use EmbeddingEngine.")
            return
            
        try:
            # Try to load the model with cache directory if specified
            kwargs = {}
            if self.cache_dir is not None:
                kwargs['cache_folder'] = self.cache_dir
            
            self.model = SentenceTransformer(self.model_name, **kwargs)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            self._load_model()
            
        if self.model is None:
            logger.error("No embedding model available")
            # Return zero vector as fallback
            return np.zeros((1, self.get_dimension()))
        
        # Handle single string or list of strings
        if isinstance(text, str):
            text = [text]
        
        # Generate embeddings
        try:
            import torch
            with torch.no_grad():
                embeddings = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero vector as fallback
            return np.zeros((len(text), self.get_dimension()))
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            Embedding dimension
        """
        if self.model is None:
            self._load_model()
        
        if self.model is None:
            # Default dimension for all-MiniLM-L6-v2
            return 384
            
        return self.model.get_sentence_embedding_dimension()
