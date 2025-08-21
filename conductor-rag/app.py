#!/usr/bin/env python3
"""
RAG Document Chat Web Application

A Gradio-based web interface for querying documents using Retrieval-Augmented Generation.
Provides an intuitive chat interface with source citations and context display.

Features:
- Real-time document search and question answering
- Source citation with page references  
- Retrieved context display for transparency
- Example queries to get users started
- Responsive web interface

Usage:
    python app.py
    
Then open your browser to the displayed URL (typically http://localhost:7860)

Requirements:
- Documents must be processed first using: python ingest.py
- Local OpenAI compatible model running on localhost:8080
"""

import gradio as gr
from typing import Tuple, List, Optional

# Import core RAG functionality
from demo import (
    create_embeddings, 
    create_llm, 
    create_qa_chain, 
    clean_response, 
    load_vectorstore,
    format_sources
)


# ============================================================================
# Application Configuration
# ============================================================================

# Gradio interface configuration
APP_TITLE = "📚 RAG Document & Code Chat"
APP_DESCRIPTION = """
Ask questions about your documents and code. The system searches through 
your ingested content and provides detailed answers with source citations.

**💡 How it works:**
1. Your question is processed and converted to embeddings
2. Relevant document sections are retrieved from the vector database
3. A language model generates an answer using the retrieved context
4. Sources are cited so you can verify the information
"""




# ============================================================================
# Core Application Logic
# ============================================================================

def initialize_rag_system():
    """
    Initialize the RAG system components.
    
    Returns:
        Configured QA chain for document retrieval and generation
        
    Raises:
        FileNotFoundError: If vector store hasn't been created
        Exception: If model loading fails
    """
    try:
        print("🔧 Initializing RAG system...")
        
        # Load models and vector store
        llm = create_llm()
        embeddings = create_embeddings() 
        vectorstore = load_vectorstore(embeddings)
        qa_chain = create_qa_chain(llm, vectorstore)
        
        print("✅ RAG system initialized successfully")
        return qa_chain
        
    except FileNotFoundError as e:
        error_msg = f"""
❌ Vector store not found: {e}

🔧 To fix this:
1. Add your files to the appropriate directories:
   - PDFs → 'pdf/' directory
   - Text/Code files → 'text/' directory
2. Run the ingestion script: python ingest.py
3. Wait for processing to complete
4. Restart this application: python app.py
"""
        print(error_msg)
        raise
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize RAG system: {e}"
        print(error_msg)
        raise


def generate_response(message: str, chat_history: List[List[str]]) -> Tuple[str, List[List[str]], str]:
    """
    Generate RAG-powered response to user query.

    Args:
        message: User's question
        chat_history: Previous conversation as list of [user_msg, bot_msg] pairs

    Returns:
        Tuple of (empty_string, updated_history, context)
        - empty_string: Clears the input box
        - updated_history: Chat history with new exchange
        - context: Retrieved document context for display
    """
    if not message.strip():
        return "", chat_history, ""
    
    try:
        # Convert Gradio history format to LangChain format
        langchain_history = []
        for exchange in chat_history:
            if len(exchange) == 2:
                user_msg, bot_msg = str(exchange[0]), str(exchange[1])
                langchain_history.append((user_msg, bot_msg))
        
        # Generate RAG response
        result = qa_chain.invoke({
            "question": message, 
            "chat_history": langchain_history
        })
        
        # Clean response and add sources
        response_text = clean_response(result["answer"])
        sources = format_sources(result.get("source_documents", []))
        
        if sources:
            full_response = response_text + sources
        else:
            full_response = response_text + "\n\n📚 Sources: No specific sources found"
        
        # Update chat history
        updated_history = chat_history + [[message, full_response]]
        
        # Extract context for display
        context = extract_context(result.get("source_documents", []))
        
        return "", updated_history, context
        
    except Exception as e:
        error_response = f"❌ Error generating response: {str(e)}"
        updated_history = chat_history + [[message, error_response]]
        return "", updated_history, "Error: Could not retrieve context"


def extract_context(source_documents: List) -> str:
    """
    Extract and format context from retrieved documents.
    
    Args:
        source_documents: List of retrieved document objects
        
    Returns:
        Formatted context string for display
    """
    if not source_documents:
        return "No context retrieved for this query."
    
    context_parts = []
    for i, doc in enumerate(source_documents[:3], 1):  # Show top 3 sources
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "unknown") 
        content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        
        context_parts.append(f"""
📄 **Source {i}:** {source} (page {page})
📝 **Content:** {content}
""")
    
    return "\n".join(context_parts)


def clear_conversation() -> Tuple[List, str]:
    """
    Clear the chat history and context display.

        Returns:
        Tuple of (empty_history, empty_context)
    """
    return [], ""


# ============================================================================
# Initialize Application
# ============================================================================

try:
    qa_chain = initialize_rag_system()
except Exception:
    # Exit gracefully if initialization fails
    print("❌ Application startup failed. Please check the error messages above.")
    exit(1)


# ============================================================================
# Gradio Interface Definition
# ============================================================================

def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="RAG Document Chat",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 100% !important;
            width: 100% !important;
        }
        .chat-container {
            height: 500px;
        }
        """
    ) as interface:
        
        # Header section
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)
        
        # Main interface layout
        with gr.Row():
            # Left column: Chat interface
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False,
                    show_copy_button=True
                )
                
                # Input area
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="💬 Your Question",
                        placeholder="Ask a question about your documents...",
                        scale=4,
                        lines=1,
                        max_lines=3
                    )
                    send_btn = gr.Button(
                        "Send 🚀", 
                        variant="primary", 
                        scale=1,
                        size="lg"
                    )
                
                # Control buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    
            # Right column: Context display
            with gr.Column(scale=1):
                context_display = gr.Textbox(
                    label="📄 Retrieved Context",
                    placeholder="Retrieved document context will appear here...",
                    interactive=False,
                    lines=20,
                    max_lines=25,
                    show_label=True
                )
        


        
        # Event handlers
        msg_input.submit(
            fn=generate_response,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, context_display],
            show_progress=True
        )
        
        send_btn.click(
            fn=generate_response,
            inputs=[msg_input, chatbot], 
            outputs=[msg_input, chatbot, context_display],
            show_progress=True
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, context_display],
            show_progress=False
        )
    
    return interface


# ============================================================================
# Application Entry Point
# ============================================================================

def main():
    """Main application entry point."""
    print("🚀 Starting RAG Document Chat Web Application")
    print("="*60)
    
    # Create interface
    interface = create_interface()
    
    # Launch configuration
    launch_config = {
        "share": False,           # Set to True to create public URL
        "server_name": "0.0.0.0", # Allow external connections
        "server_port": 7860,      # Default Gradio port
        "show_error": True,       # Show errors in interface
        "quiet": False,           # Show startup logs
        "favicon_path": None,     # Could add custom favicon
    }
    
    print(f"🌐 Launching web interface...")
    print(f"📱 Open your browser to: http://localhost:{launch_config['server_port']}")
    print(f"🛑 Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        interface.launch(**launch_config)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()