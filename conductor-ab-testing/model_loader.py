"""
Model loader module for A/B Testing Application

This module centralizes the loading of ML models to avoid circular imports
and provides efficient model management for semantic similarity calculations.

The module uses SentenceTransformers for computing semantic similarity between
model responses, which is essential for comprehensive A/B testing analysis.

Author: Arcee AI Team
License: MIT
"""

import logging
from typing import Optional
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the embedding model name as a constant
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Initialize the model as None (lazy loading)
_model: Optional[SentenceTransformer] = None


def get_model() -> Optional[SentenceTransformer]:
    """
    Get the SentenceTransformer model, loading it if necessary.
    
    This function implements lazy loading of the embedding model to avoid
    unnecessary memory usage and startup time. The model is loaded only
    when first requested and cached for subsequent calls.
    
    Returns:
        Optional[SentenceTransformer]: The loaded SentenceTransformer model,
            or None if loading fails.
    
    Note:
        The function uses the BAAI/bge-large-en-v1.5 model which provides
        excellent performance for semantic similarity calculations in English.
        The model is cached globally to avoid reloading on subsequent calls.
    """
    global _model
    
    if _model is None:
        try:
            logger.info(f"Loading {EMBEDDING_MODEL} embedding model...")
            _model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Successfully loaded {EMBEDDING_MODEL} embedding model")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            _model = None
            return None
    
    return _model


def clear_model() -> None:
    """
    Clear the cached model to free memory.
    
    This function can be used to explicitly free the memory used by the
    cached SentenceTransformer model. Useful for memory management in
    long-running applications.
    
    Note:
        After calling this function, the next call to get_model() will
        reload the model from scratch.
    """
    global _model
    
    if _model is not None:
        logger.info("Clearing cached embedding model")
        _model = None
    else:
        logger.debug("No cached model to clear")
