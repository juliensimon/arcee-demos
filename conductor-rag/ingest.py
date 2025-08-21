#!/usr/bin/env python3
"""
RAG Document Ingestion Tool

A comprehensive tool for processing documents and creating vector embeddings for
Retrieval-Augmented Generation (RAG) applications.

Features:
- Multi-format document support (PDF, text, markdown, code files)
- Automatic device detection (CUDA, MPS, CPU)
- Progress tracking and batched processing
- Incremental updates for new documents
- Vector store analysis and statistics

Usage:
    python ingest.py                 # Run full ingestion
    python ingest.py analyze         # Analyze existing vector store
    python ingest.py --help          # Show detailed help
"""

import argparse
import glob
import json
import os
import sys
import warnings
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._fields")
warnings.filterwarnings("ignore", message=".*Defaulting to English.*")
warnings.filterwarnings("ignore", category=UserWarning, module="unstructured")

# Additional suppression for unstructured library language detection
import logging
logging.getLogger("unstructured").setLevel(logging.ERROR)

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader
)
from langchain_huggingface import HuggingFaceEmbeddings

# Global configuration variables
config: Optional[Dict[str, Any]] = None
CHROMA_PATH: Optional[str] = None
PDF_PATH: Optional[str] = None
TEXT_PATH: Optional[str] = None


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        SystemExit: If configuration file is not found
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found")
        print("Please ensure config.json exists in the current directory")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def initialize_paths(config_data: Dict[str, Any]) -> None:
    """
    Initialize global path variables from configuration.
    
    Args:
        config_data: Configuration dictionary
    """
    global config, CHROMA_PATH, PDF_PATH, TEXT_PATH
    config = config_data
    CHROMA_PATH = config["paths"]["vectorstore"]
    PDF_PATH = config["paths"]["pdf"]
    TEXT_PATH = config["paths"]["text"]


# ============================================================================
# Device Detection and Model Loading
# ============================================================================

def detect_optimal_device() -> str:
    """
    Detect the optimal device for model inference.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            print(f"🚀 Auto-detected CUDA GPU: {device_name}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("🍎 Auto-detected Apple Silicon MPS GPU")
            return "mps"
        else:
            print("💻 Auto-detected CPU (no GPU available)")
            return "cpu"
            
    except ImportError:
        print("Warning: PyTorch not available, using CPU")
        return "cpu"


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize the embedding model with automatic device detection and fallback.
    
    Returns:
        Configured HuggingFaceEmbeddings instance
        
    Raises:
        RuntimeError: If model loading fails on all devices
    """
    if not config:
        raise RuntimeError("Configuration not loaded. Call initialize_paths() first.")
    
    embeddings_config = config["embeddings"]
    model_name = embeddings_config["model_name"]
    device = detect_optimal_device()
    
    # Try MPS with fallback to CPU for Apple Silicon compatibility
    if device == "mps":
        try:
            print("🔄 Attempting to load model with MPS...")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "mps", "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": embeddings_config["normalize_embeddings"]},
            )
        except (ValueError, NotImplementedError, RuntimeError) as e:
            error_keywords = ["meta tensor", "meta device", "Cannot copy out of meta tensor"]
            if any(keyword in str(e) for keyword in error_keywords):
                print("⚠️  MPS failed with meta tensor error, falling back to CPU")
                device = "cpu"
            else:
                raise
    
    # Use CUDA or CPU (or fallback from MPS)
    print(f"🔄 Loading model with {device.upper()}: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": embeddings_config["normalize_embeddings"]},
    )


# ============================================================================
# Text Processing
# ============================================================================

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with configurable settings.
    
    Returns:
        Configured RecursiveCharacterTextSplitter instance
    """
    if not config:
        raise RuntimeError("Configuration not loaded")
    
    text_config = config["text_splitting"]
    return RecursiveCharacterTextSplitter(
        chunk_size=text_config["chunk_size"],
        chunk_overlap=text_config["chunk_overlap"],
        length_function=len,
        add_start_index=True,
    )


def filter_metadata(doc) -> bool:
    """
    Filter out unwanted document sections.
    
    Args:
        doc: Document object with metadata
        
    Returns:
        True if document should be kept, False if filtered out
    """
    skip_sections = {"references", "acknowledgments", "appendix"}
    section = doc.metadata.get("section", "").lower()
    return not any(section_keyword in section for section_keyword in skip_sections)


def process_documents(documents: List, text_splitter: RecursiveCharacterTextSplitter) -> List:
    """
    Process documents into filtered chunks.
    
    Args:
        documents: List of loaded documents
        text_splitter: Text splitter instance
        
    Returns:
        List of filtered document chunks
    """
    if not documents:
        return []
    
    print(f"Processing {len(documents)} documents...")
    
    # Split documents into chunks
    print("  Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"    Created {len(chunks)} chunks")
    
    # Filter chunks with progress bar
    print("  Filtering chunks...")
    filtered_chunks = [
        chunk for chunk in tqdm(chunks, desc="Filtering", unit="chunk")
        if filter_metadata(chunk)
    ]
    
    print(f"    Kept {len(filtered_chunks)} chunks after filtering")
    return filtered_chunks


# ============================================================================
# File Discovery
# ============================================================================

def get_pdf_files() -> List[str]:
    """
    Get list of PDF files from the configured directory.
    
    Returns:
        List of PDF file paths
    """
    if not os.path.exists(PDF_PATH):
        os.makedirs(PDF_PATH, exist_ok=True)
        return []
    return list(glob.glob(os.path.join(PDF_PATH, "*.pdf")))


def get_text_files() -> List[str]:
    """
    Get list of text files from the configured directory.
    
    Returns:
        List of text file paths
    """
    if not os.path.exists(TEXT_PATH):
        os.makedirs(TEXT_PATH, exist_ok=True)
        return []
    
    text_extensions = config["file_types"]["text_extensions"]
    text_files = []
    
    for ext in text_extensions:
        # Search for both lowercase and uppercase extensions
        text_files.extend(glob.glob(os.path.join(TEXT_PATH, f"*{ext}")))
        text_files.extend(glob.glob(os.path.join(TEXT_PATH, f"*{ext.upper()}")))
    
    return text_files


def get_all_files() -> List[str]:
    """
    Get list of all supported files (PDF and text).
    
    Returns:
        Combined list of all file paths
    """
    return get_pdf_files() + get_text_files()


# ============================================================================
# Document Loading
# ============================================================================

def create_document_loaders() -> List:
    """
    Create document loaders for different file types.
    
    Returns:
        List of configured document loaders
    """
    loaders = []
    
    # PDF loader
    if os.path.exists(PDF_PATH):
        pdf_loader = DirectoryLoader(
            PDF_PATH, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        loaders.append(pdf_loader)
    
    # Text file loaders
    if os.path.exists(TEXT_PATH):
        text_extensions = config["file_types"]["text_extensions"]
        
        # Create glob patterns for all text extensions
        text_patterns = []
        for ext in text_extensions:
            text_patterns.extend([f"**/*{ext}", f"**/*{ext.upper()}"])
        
        # General text loader
        text_loader = DirectoryLoader(
            TEXT_PATH, 
            glob=text_patterns, 
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        loaders.append(text_loader)
        
        # Specialized markdown loader
        md_patterns = ["**/*.md", "**/*.markdown", "**/*.MD", "**/*.MARKDOWN"]
        md_loader = DirectoryLoader(
            TEXT_PATH,
            glob=md_patterns,
            loader_cls=UnstructuredMarkdownLoader
        )
        loaders.append(md_loader)
    
    return loaders


def load_documents_with_progress(loaders: List) -> List:
    """
    Load documents from all loaders with progress tracking.
    
    Args:
        loaders: List of document loaders
        
    Returns:
        List of loaded documents
    """
    all_documents = []
    
    print("Loading documents...")
    for loader in loaders:
        loader_name = loader.__class__.__name__
        loader_path = getattr(loader, 'path', 'unknown')
        loader_glob = getattr(loader, 'glob', 'unknown')
        
        # Create a descriptive name for the loader
        if 'PDF' in loader_name:
            description = f"PDFs from {loader_path}/"
        elif 'Markdown' in loader_name:
            description = f"Markdown files from {loader_path}/"
        elif 'Text' in loader_name:
            description = f"Text files from {loader_path}/"
        else:
            description = f"{loader_name} from {loader_path}/"
        
        print(f"  Loading {description}")
        print(f"    Pattern: {loader_glob}")
        
        try:
            documents = loader.load()
            all_documents.extend(documents)
            print(f"    ✅ Loaded {len(documents)} documents")
        except Exception as e:
            print(f"    ❌ Warning: Error loading: {e}")
    
    return all_documents


# ============================================================================
# Vector Store Management
# ============================================================================

def load_existing_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load an existing vector store.
    
    Args:
        embeddings: Embedding model instance
        
    Returns:
        Loaded Chroma vector store
        
    Raises:
        FileNotFoundError: If vector store doesn't exist
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_PATH}. "
            "Please run 'python ingest.py' first to process your documents."
        )
    
    print("📚 Loading existing Chroma database...")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


def update_vectorstore_with_progress(
    vectorstore: Chroma, 
    chunks: List, 
    batch_size: int = 10
) -> None:
    """
    Add document chunks to vector store with progress tracking.
    
    Args:
        vectorstore: Chroma vector store instance
        chunks: List of document chunks to add
        batch_size: Number of chunks to process per batch
    """
    if not chunks:
        return
    
    print("Adding documents to vector store...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Adding chunks", unit="batch"):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)


def create_vectorstore_with_progress(
    embeddings: HuggingFaceEmbeddings, 
    chunks: List
) -> Chroma:
    """
    Create a new vector store with progress tracking for large datasets.
    
    Args:
        embeddings: Embedding model instance
        chunks: List of document chunks
        
    Returns:
        Created Chroma vector store
    """
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    print("Creating vector embeddings and storing in database...")
    
    # Use batched processing for large datasets
    if len(chunks) > 50:
        batch_size = 25
        print(f"Processing {len(chunks)} chunks in batches of {batch_size}...")
        
        # Create initial vector store with first batch
        first_batch = chunks[:batch_size]
        vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
        )
        
        # Add remaining batches with progress
        remaining_chunks = chunks[batch_size:]
        if remaining_chunks:
            update_vectorstore_with_progress(vectorstore, remaining_chunks, batch_size)
        
        return vectorstore
    else:
        # Create normally for small datasets
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
        )


def handle_existing_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load existing vector store and update with new documents if needed.
    
    Args:
        embeddings: Embedding model instance
        
    Returns:
        Updated vector store
    """
    vectorstore = load_existing_vectorstore(embeddings)
    
    # Check for new files
    all_files = get_all_files()
    if not all_files:
        print("No supported files found in directories.")
        print(f"Please add files to '{PDF_PATH}' (PDFs) or '{TEXT_PATH}' (text files)")
        sys.exit(1)
    
    # Get already processed files
    collection = vectorstore.get()
    processed_files = {
        meta.get("source") for meta in collection.get("metadatas", [])
        if meta and meta.get("source")
    }
    
    # Find new files
    new_files = [f for f in all_files if f not in processed_files]
    
    if new_files:
        print(f"Found {len(new_files)} new files to process...")
        
        # Load and process new documents
        loaders = create_document_loaders()
        all_documents = load_documents_with_progress(loaders)
        new_documents = [
            doc for doc in all_documents 
            if doc.metadata.get("source") not in processed_files
        ]
        
        # Process and add new chunks
        if new_documents:
            filtered_chunks = process_documents(new_documents, get_text_splitter())
            if filtered_chunks:
                update_vectorstore_with_progress(vectorstore, filtered_chunks)
                print("Database updated successfully!")
    else:
        print("No new files to process.")
    
    return vectorstore


def create_new_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Create a new vector store from all available documents.
    
    Args:
        embeddings: Embedding model instance
        
    Returns:
        Created vector store
    """
    print("Creating new Chroma database...")
    
    # Check for files
    all_files = get_all_files()
    if not all_files:
        print(f"No supported files found!")
        print(f"Please add files to:")
        print(f"  - '{PDF_PATH}' directory (for PDF files)")
        print(f"  - '{TEXT_PATH}' directory (for text files)")
        sys.exit(1)
    
    print(f"Found {len(all_files)} files to process...")
    print("(This may take a while for large document collections)")
    
    # Load and process all documents
    loaders = create_document_loaders()
    all_documents = load_documents_with_progress(loaders)
    filtered_chunks = process_documents(all_documents, get_text_splitter())
    
    return create_vectorstore_with_progress(embeddings, filtered_chunks)


def load_or_create_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load existing vector store or create a new one.
    
    Args:
        embeddings: Embedding model instance
        
    Returns:
        Vector store instance
    """
    if os.path.exists(CHROMA_PATH):
        return handle_existing_vectorstore(embeddings)
    return create_new_vectorstore(embeddings)


# ============================================================================
# Analysis and Statistics
# ============================================================================

def analyze_vectorstore(vectorstore: Chroma) -> None:
    """
    Analyze vector store and display statistics about indexed files.
    
    Args:
        vectorstore: Chroma vector store to analyze
    """
    collection = vectorstore.get()
    if not collection.get("documents"):
        print("Vector store is empty.")
        return
    
    # Collect file statistics
    file_stats = {}
    total_chunks = len(collection["documents"])
    
    for metadata in collection.get("metadatas", []):
        if metadata and metadata.get("source"):
            source = metadata["source"]
            file_ext = os.path.splitext(source)[1].lower()
            
            if file_ext not in file_stats:
                file_stats[file_ext] = {"files": set(), "chunks": 0}
            
            file_stats[file_ext]["files"].add(source)
            file_stats[file_ext]["chunks"] += 1
    
    # Display statistics
    print(f"\n{'='*50}")
    print(f"Vector Store Statistics")
    print(f"{'='*50}")
    print(f"Total document chunks: {total_chunks:,}")
    print(f"Unique files indexed: {sum(len(stats['files']) for stats in file_stats.values())}")
    print(f"\nBreakdown by file type:")
    
    # Sort by chunk count (descending)
    sorted_stats = sorted(file_stats.items(), key=lambda x: x[1]["chunks"], reverse=True)
    
    for file_ext, stats in sorted_stats:
        num_files = len(stats["files"])
        num_chunks = stats["chunks"]
        percentage = (num_chunks / total_chunks) * 100
        
        print(f"  {file_ext:>8}: {num_files:>3} files, {num_chunks:>4} chunks ({percentage:>5.1f}%)")
        
        # Show sample files
        file_list = sorted(list(stats["files"]))
        for file_path in file_list[:3]:
            filename = os.path.basename(file_path)
            print(f"           - {filename}")
        if len(file_list) > 3:
            print(f"           ... and {len(file_list) - 3} more files")


# ============================================================================
# Command Line Interface
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="RAG Document Ingestion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                    # Run full ingestion process
  python ingest.py analyze            # Analyze existing vector store
  python ingest.py --config custom.json  # Use custom config file

Supported file types:
  PDF files: .pdf
  Text files: .txt, .md, .markdown
  Code files: .py, .js, .ts, .java, .cpp, .c, .go, .rs, .php, .rb, .swift, .kt, .scala
  Scripts: .sh, .bash, .zsh, .fish, .sql
  Web: .html, .css, .scss, .sass  
  Config: .json, .xml, .yaml, .yml, .toml, .ini, .cfg, .conf
  Hardware: .vhd, .vhdl, .v, .sv, .svh
  Logs: .log
        """
    )
    
    parser.add_argument(
        "action",
        nargs="?",
        default="ingest",
        choices=["ingest", "analyze"],
        help="Action to perform (default: ingest)"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of all files (ignores existing vector store)"
    )
    
    return parser


# ============================================================================
# Main Functions
# ============================================================================

def run_analysis(config_path: str) -> None:
    """
    Run analysis on existing vector store.
    
    Args:
        config_path: Path to configuration file
        """
    try:
        config_data = load_config(config_path)
        initialize_paths(config_data)
        
        embeddings = create_embeddings()
        vectorstore = load_existing_vectorstore(embeddings)
        analyze_vectorstore(vectorstore)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n🔧 To fix this:")
        print(f"   1. Add your files to the 'pdf' and/or 'text' directories")
        print(f"   2. Run 'python ingest.py' to process the documents")
    except Exception as e:
        print(f"❌ Error analyzing vector store: {e}")
        sys.exit(1)


def run_ingestion(config_path: str, force: bool = False) -> None:
    """
    Run the main ingestion process.
    
    Args:
        config_path: Path to configuration file
        force: Whether to force re-ingestion of all files
    """
    print("🚀 Starting RAG document ingestion process...")
    print("="*60)
    
    # Load configuration and initialize paths
    config_data = load_config(config_path)
    initialize_paths(config_data)
    
    # Display configuration info
    text_extensions = config["file_types"]["text_extensions"]
    print(f"\nSupported file types:")
    print(f"  - PDF files: {PDF_PATH}/")
    print(f"  - Text files: {TEXT_PATH}/")
    print(f"    Extensions: {', '.join(text_extensions[:10])}")
    if len(text_extensions) > 10:
        print(f"    ... and {len(text_extensions) - 10} more")
    
    # Handle force re-ingestion
    if force:
        print("\n⚠️  Force flag detected - this feature is not yet implemented")
        print("📝 TODO: Add logic to clear existing vector store")
    
    try:
        # Initialize embeddings and process documents
        embeddings = create_embeddings()
        vectorstore = load_or_create_vectorstore(embeddings)
        
        # Show final statistics
        analyze_vectorstore(vectorstore)
        
        print(f"\nVector store location: {os.path.abspath(CHROMA_PATH)}")
        print("\n✅ Ingestion complete! You can now run the chatbot with: python app.py")
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point for the application."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.action == "analyze":
        run_analysis(args.config)
    elif args.action == "ingest":
        run_ingestion(args.config, args.force)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()