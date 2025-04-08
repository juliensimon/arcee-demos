"""
Model loader module for A/B Testing Application

This module centralizes the loading of ML models to avoid circular imports.
"""

import logging
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the embedding model name as a constant
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Initialize the model as None
model = None


def get_model():
    """
    Get the SentenceTransformer model, loading it if necessary.

    Returns:
        SentenceTransformer: The loaded model
    """
    global model
    if model is None:
        try:
            model = SentenceTransformer(EMBEDDING_MODEL)
            logging.info(f"Loaded {EMBEDDING_MODEL} embedding model")
        except Exception as e:
            logging.error(f"Error loading sentence transformer model: {e}")
            model = None
    return model
