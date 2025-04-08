import math
import logging
from collections import Counter
from typing import Set, Dict, List, Optional
from sentence_transformers import SentenceTransformer, util
from model_loader import get_model
import uuid


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    Jaccard similarity is defined as the size of the intersection divided by the size of the union of two sets.
    This implementation treats each text as a set of unique words.

    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare

    Returns:
        float: Jaccard similarity score between 0 and 1, where 1 means identical sets of words
    """
    # Convert to sets of words (lowercase and split by whitespace)
    words1: Set[str] = set(text1.lower().split())
    words2: Set[str] = set(text2.lower().split())

    # Calculate intersection (words that appear in both texts)
    intersection: Set[str] = words1.intersection(words2)
    # Calculate union (all unique words from both texts)
    union: Set[str] = words1.union(words2)

    # Return Jaccard similarity (intersection size / union size)
    return len(intersection) / len(union)


def cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using bag-of-words approach.

    Cosine similarity measures the cosine of the angle between two non-zero vectors.
    This implementation represents texts as word frequency vectors.

    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare

    Returns:
        float: Cosine similarity score between 0 and 1, where 1 means identical direction
    """
    # Create word frequency dictionaries (term frequency vectors)
    vec1: Dict[str, int] = Counter(text1.lower().split())
    vec2: Dict[str, int] = Counter(text2.lower().split())

    # Find common words between both texts
    intersection: Set[str] = set(vec1.keys()) & set(vec2.keys())

    # Calculate dot product of the two vectors (sum of products of common word frequencies)
    dot_product: int = sum(vec1[x] * vec2[x] for x in intersection)

    # Calculate magnitudes (Euclidean norms) of each vector
    magnitude1: float = math.sqrt(sum(val**2 for val in vec1.values()))
    magnitude2: float = math.sqrt(sum(val**2 for val in vec2.values()))

    # Handle zero division case (when either text has no words)
    if magnitude1 * magnitude2 == 0:
        return 0

    # Return cosine similarity (dot product / product of magnitudes)
    return dot_product / (magnitude1 * magnitude2)


def levenshtein_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity based on Levenshtein (edit) distance between two texts.

    Levenshtein distance counts the minimum number of single-character operations
    (insertions, deletions, substitutions) required to change one text into another.
    This implementation converts the distance to a similarity score.

    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare

    Returns:
        float: Similarity score between 0 and 1, where 1 means identical texts
    """
    # Calculate Levenshtein distance using dynamic programming
    m, n = len(text1), len(text2)
    # Initialize distance matrix with zeros
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming empty string to string of length i requires i insertions
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If characters match, no additional cost
            cost: int = 0 if text1[i - 1] == text2[j - 1] else 1
            # Choose minimum cost operation (deletion, insertion, or substitution)
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    # Convert distance to similarity (0 to 1 scale)
    max_len: int = max(m, n)
    if max_len == 0:
        return 1.0  # Both strings are empty

    # Normalize by dividing by maximum possible distance and subtract from 1
    return 1 - (dp[m][n] / max_len)


def semantic_similarity(
    text1: str, text2: str, model: Optional[SentenceTransformer] = None
) -> float:
    """
    Calculate semantic similarity between two texts using sentence transformers.

    This implementation uses a pre-trained transformer model to generate embeddings
    and computes the cosine similarity between them. The similarity score indicates
    how semantically similar the meanings of the two texts are.

    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        model (SentenceTransformer, optional): Pre-loaded SentenceTransformer model.
            If None, will raise an error as a model must be provided

    Returns:
        float: Semantic similarity score between 0 and 1, where 1 means semantically identical
    """
    if not text1 or not text2:
        return 0.0

    try:
        # Ensure model is provided
        if model is None:
            raise ValueError("A SentenceTransformer model must be provided")

        # Generate embeddings for both texts
        embeddings = model.encode([text1, text2])

        # Calculate cosine similarity between embeddings
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

        return similarity

    except Exception as e:
        logging.error(f"Error calculating semantic similarity: {e}")
        return 0.0


# Example usage
if __name__ == "__main__":
    # Example usage of similarity functions
    sample1 = "The quick brown fox jumps over the lazy dog."
    sample2 = "A fast brown fox leaps over a sleepy dog."
    sample3 = "The quick brown fox jumps over the lazy dog."
    sample4 = "The speedy brown fox jumps over the lazy dog."
    sample5 = "The weather is nice today."

    # Jaccard similarity comparisons
    print("Jaccard Similarity:")
    print(f"Text 1 vs Text 2: {jaccard_similarity(sample1, sample2):.4f}")
    print(f"Text 1 vs Text 3: {jaccard_similarity(sample1, sample3):.4f}")
    print(f"Text 1 vs Text 4: {jaccard_similarity(sample1, sample4):.4f}")
    print(f"Text 1 vs Text 5: {jaccard_similarity(sample1, sample5):.4f}")

    # Cosine similarity comparisons
    print("\nCosine Similarity:")
    print(f"Text 1 vs Text 2: {cosine_similarity(sample1, sample2):.4f}")
    print(f"Text 1 vs Text 3: {cosine_similarity(sample1, sample3):.4f}")
    print(f"Text 1 vs Text 4: {cosine_similarity(sample1, sample4):.4f}")
    print(f"Text 1 vs Text 5: {cosine_similarity(sample1, sample5):.4f}")

    # Levenshtein similarity comparisons
    print("\nLevenshtein Similarity:")
    print(f"Text 1 vs Text 2: {levenshtein_similarity(sample1, sample2):.4f}")
    print(f"Text 1 vs Text 3: {levenshtein_similarity(sample1, sample3):.4f}")
    print(f"Text 1 vs Text 4: {levenshtein_similarity(sample1, sample4):.4f}")
    print(f"Text 1 vs Text 5: {levenshtein_similarity(sample1, sample5):.4f}")

    # Semantic similarity comparisons
    print("\nSemantic Similarity:")
    try:
        # Get the model from the model_loader
        model = get_model()

        if model is None:
            print("Warning: Model not loaded. Skipping semantic similarity examples.")
        else:
            print(
                f"Text 1 vs Text 2: {semantic_similarity(sample1, sample2, model):.4f}"
            )
            print(
                f"Text 1 vs Text 3: {semantic_similarity(sample1, sample3, model):.4f}"
            )
            print(
                f"Text 1 vs Text 4: {semantic_similarity(sample1, sample4, model):.4f} (should be high - similar meaning)"
            )
            print(
                f"Text 1 vs Text 5: {semantic_similarity(sample1, sample5, model):.4f} (should be low - different meaning)"
            )
    except Exception as e:
        print(f"Error running semantic similarity examples: {e}")

    # Demonstrate with identical texts
    print("\nIdentical text comparison:")
    print(f"Jaccard: {jaccard_similarity(sample1, sample1):.4f}")
    print(f"Cosine: {cosine_similarity(sample1, sample1):.4f}")
    print(f"Levenshtein: {levenshtein_similarity(sample1, sample1):.4f}")
    try:
        if model is not None:
            print(f"Semantic: {semantic_similarity(sample1, sample1, model):.4f}")
    except Exception as e:
        print(f"Error in semantic similarity for identical texts: {e}")

    # Analysis of different similarity measures
    print("\nAnalysis:")
    print(
        "- Jaccard similarity focuses on word overlap, ignoring word order and semantics"
    )
    print(
        "- Cosine similarity considers word frequencies, still missing semantic relationships"
    )
    print("- Levenshtein similarity works at character level, sensitive to small edits")
    print(
        "- Semantic similarity captures meaning using neural embeddings, best for meaning comparison"
    )
