#!/usr/bin/env python3
"""
RAG Demo Module

Utilities for document querying using Retrieval-Augmented Generation.
This module provides core functionality for:
- Loading pre-processed vector embeddings
- Setting up language models and QA chains  
- Handling RAG-powered responses with source citations
- Command-line testing interface

Usage:
    python demo.py  # Run a sample query
    
Or import functions for use in other applications:
    from demo import create_embeddings, create_llm, create_qa_chain
"""

import json
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._fields"
)

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# ============================================================================
# Configuration Management
# ============================================================================


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file with fallback defaults.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary

    Note:
        Uses fallback configuration if file is not found.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: config.json not found, using minimal defaults")
        return {
            "embeddings": {
                "model_name": "BAAI/bge-small-en-v1.5",
                "normalize_embeddings": True,
            },
            "paths": {"vectorstore": "vectorstore", "pdf": "pdf"},
            "retrieval": {"num_chunks": 3},
        }
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        raise


# Load global configuration
config = load_config()
CHROMA_PATH = config["paths"]["vectorstore"]


# ============================================================================
# Text Cleaning Utilities
# ============================================================================


def clean_response(text: str) -> str:
    """
    Clean model response by removing special tokens and formatting artifacts.

    Args:
        text: Raw response text from the model

    Returns:
        Cleaned response text with special tokens removed

    Note:
        Removes common LLM special tokens like <|im_end|>, <|endoftext|>, etc.
    """
    if not text:
        return text

    # Common special tokens to remove
    special_tokens = [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "<|start|>",
    ]

    cleaned_text = text
    for token in special_tokens:
        cleaned_text = cleaned_text.replace(token, "")

    # Clean up whitespace
    return cleaned_text.strip()


# ============================================================================
# Model Loading
# ============================================================================


def create_llm(streaming: bool = None) -> ChatOpenAI:
    """
    Initialize OpenAI compatible language model using configuration.

    Args:
        streaming: Whether to enable response streaming (overrides config if provided)

    Returns:
        Configured ChatOpenAI instance

    Note:
        Uses configuration from config.json for endpoint settings
    """
    if not config:
        raise RuntimeError("Configuration not loaded")

    llm_config = config.get("llm", {})

    # Use provided streaming parameter or fall back to config
    use_streaming = (
        streaming if streaming is not None else llm_config.get("streaming", False)
    )

    return ChatOpenAI(
        model=llm_config.get("model_name", "local-model"),
        base_url=llm_config.get("base_url", "http://localhost:8080/v1"),
        api_key=llm_config.get("api_key", "dummy-key"),
        streaming=use_streaming,
        temperature=llm_config.get("temperature", 0.1),
        max_tokens=llm_config.get("max_tokens", 2048),
    )


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize embedding model for chat/inference (always uses CPU).

    Returns:
        Configured HuggingFaceEmbeddings instance

    Raises:
        RuntimeError: If model loading fails

    Note:
        Always uses CPU for chat to ensure consistent performance and compatibility.
        For ingestion with auto-detection, use the create_embeddings function in ingest.py
    """
    if not config:
        raise RuntimeError("Configuration not loaded")

    embeddings_config = config["embeddings"]
    model_name = embeddings_config["model_name"]

    # Always use CPU for chat/inference
    print(f"🔄 Loading embedding model for chat (CPU): {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={
            "normalize_embeddings": embeddings_config["normalize_embeddings"]
        },
    )


# ============================================================================
# Vector Store Management
# ============================================================================


def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load existing vector store for document querying.

    Args:
        embeddings: Embedding model instance

    Returns:
        Loaded Chroma vector store

    Raises:
        FileNotFoundError: If vector store doesn't exist

    Note:
        Vector store must be created first using ingest.py
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"📂 Vector store not found at '{CHROMA_PATH}'\n"
            "Please run 'python ingest.py' first to process your documents."
        )

    print(f"📚 Loading vector database from: {CHROMA_PATH}")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


# ============================================================================
# Question-Answering Chain Setup
# ============================================================================


def create_qa_chain(
    llm: ChatOpenAI, vectorstore: Chroma
) -> ConversationalRetrievalChain:
    """
    Create the RAG question-answering chain.

    Args:
        llm: Language model instance
        vectorstore: Vector store for document retrieval

    Returns:
        Configured ConversationalRetrievalChain for RAG

    Note:
        Uses similarity search with configurable number of documents for context
        (controlled by config.retrieval.num_chunks)
    """
    # Get prompt configuration from config
    prompt_config = config.get("prompt", {})
    role = prompt_config.get(
        "role",
        "You are a helpful AI assistant that answers questions based on provided context and your knowledge.",
    )
    instructions = prompt_config.get(
        "instructions",
        [
            "Use the provided context to answer the question accurately",
            "If the context doesn't contain enough information, say so clearly",
            "Provide specific details and examples when available",
            "Be concise but comprehensive",
            "Cite relevant information from the context",
        ],
    )

    # Build instructions string
    instructions_text = "\n".join([f"- {instruction}" for instruction in instructions])

    # Custom prompt template using configurable settings
    prompt_template = f"""{role}

Context from documents:
{{context}}

Previous conversation:
{{chat_history}}

Question: {{question}}

Instructions:
{instructions_text}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Get number of chunks from config
    num_chunks = config.get("retrieval", {}).get("num_chunks", 3)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_chunks},  # Retrieve top N most relevant documents
        ),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        chain_type="stuff",  # Stuff all retrieved docs into the prompt
        verbose=False,  # Set to True for debugging
    )


# ============================================================================
# Response Generation and Display
# ============================================================================


def format_sources(source_documents: List) -> str:
    """
    Format source documents for display.

    Args:
        source_documents: List of retrieved document objects

    Returns:
        Formatted string with unique source citations
    """
    if not source_documents:
        return ""

    sources = []
    seen_sources = set()

    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "unknown")
        source_key = f"{source}:{page}"

        if source_key not in seen_sources:
            # Clean up source path for display
            display_source = os.path.basename(source) if source != "Unknown" else source
            sources.append(f"  📄 {display_source}, page {page}")
            seen_sources.add(source_key)

    return "\n\n📚 Sources:\n" + "\n".join(sources) if sources else ""


def get_rag_response(
    qa_chain: ConversationalRetrievalChain,
    query: str,
    chat_history: List[Tuple[str, str]],
) -> None:
    """
    Generate and display RAG-powered response.

    Args:
        qa_chain: The QA chain instance
        query: User's question
        chat_history: Previous conversation history as list of (human, ai) tuples

    Note:
        Displays complete response with source citations
    """
    print("\n" + "=" * 60)
    print("🤖 RAG Response")
    print("=" * 60)

    try:
        # Get response from QA chain
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})

        # Clean and display answer
        answer = result.get("answer", "❌ No answer found.")
        cleaned_answer = clean_response(answer)

        print(f"\n💬 Question: {query}")
        print(f"\n✨ Answer: {cleaned_answer}")

        # Display sources
        sources = format_sources(result.get("source_documents", []))
        if sources:
            print(sources)
        else:
            print("\n📚 Sources: No specific sources found")

        print("\n" + "=" * 60)

    except (KeyboardInterrupt, ConnectionError) as e:
        print(f"\n❌ Error generating response: {str(e)}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")


# ============================================================================
# Main Demo Function
# ============================================================================


def main() -> None:
    """
    Main demo function for testing RAG functionality.

    Demonstrates:
    - Loading vector store
    - Creating QA chain
    - Generating RAG response
    - Displaying results with sources
    """
    print("🚀 Starting RAG Demo")
    print("=" * 60)

    try:
        # Initialize components
        print("🔧 Initializing components...")
        llm = create_llm()  # Uses streaming setting from config
        embeddings = create_embeddings()
        vectorstore = load_vectorstore(embeddings)
        qa_chain = create_qa_chain(llm, vectorstore)

        # Test query
        chat_history = []
        query = "What are the main topics covered in the documentation?"

        print(f"✅ Initialization complete!")

        # Generate response
        get_rag_response(qa_chain, query, chat_history)

        print(f"\n🎉 Demo completed successfully!")
        print(f"💡 Try running 'python app.py' for the web interface")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n🔧 To fix this:")
        print(f"   1. Add your files to the 'pdf' and/or 'text' directories")
        print(f"   2. Run 'python ingest.py' to process the documents")
        print(f"   3. Then run this script again")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
