"""
Cyberpunk-styled Technical Document AI Interface
Enhanced with neural processing aesthetics and advanced functionality
"""
import warnings
import os

# Suppress common harmless warnings
warnings.filterwarnings("ignore", message="PyPDF2 is deprecated")
warnings.filterwarnings("ignore", message=".*applymap.*deprecated.*")
os.environ['JUPYTER_PLATFORM_DIRS'] = '1'

import streamlit as st
import sys
from pathlib import Path
import tempfile
import uuid
from datetime import datetime
import json
import time
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from main import TechnicalDocumentAI
from loguru import logger

# Configure Streamlit page
st.set_page_config(
    page_title="NEURAL DOCS - Technical AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for cyberpunk theme
def load_cyberpunk_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Courier+Prime:wght@400;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a0033 25%, #000000 50%, #001a33 75%, #000000 100%);
        background-attachment: fixed;
        color: #00ffff;
        font-family: 'Courier Prime', monospace;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Cyberpunk title */
    .cyber-title {
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00ffff, #ff00ff, #ffff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
        margin-bottom: 20px;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(0, 255, 255, 0.5); }
        to { text-shadow: 0 0 30px rgba(255, 0, 255, 0.8); }
    }
    
    .cyber-subtitle {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        text-align: center;
        color: #00ffff;
        margin-bottom: 30px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Neon box styles */
    .neon-box {
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 25px;
        background: rgba(0, 0, 0, 0.8);
        box-shadow: 
            0 0 20px rgba(0, 255, 255, 0.3),
            inset 0 0 20px rgba(0, 255, 255, 0.1);
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }
    
    .neon-box:hover {
        border-color: #ff00ff;
        box-shadow: 
            0 0 30px rgba(255, 0, 255, 0.5),
            inset 0 0 20px rgba(255, 0, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .neon-box-green {
        border: 2px solid #00ff00;
        border-radius: 15px;
        padding: 25px;
        background: rgba(0, 0, 0, 0.8);
        box-shadow: 
            0 0 20px rgba(0, 255, 0, 0.3),
            inset 0 0 20px rgba(0, 255, 0, 0.1);
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }
    
    .neon-box-purple {
        border: 2px solid #ff00ff;
        border-radius: 15px;
        padding: 25px;
        background: rgba(0, 0, 0, 0.8);
        box-shadow: 
            0 0 20px rgba(255, 0, 255, 0.3),
            inset 0 0 20px rgba(255, 0, 255, 0.1);
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }
    
    /* File upload styling */
    .stFileUploader > div > div {
        background: rgba(0, 0, 0, 0.9);
        border: 2px dashed #00ffff;
        border-radius: 10px;
        padding: 20px;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #ff00ff;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.7);
    }
    
    /* Chat messages */
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 15px;
        font-family: 'Courier Prime', monospace;
    }
    
    .user-message {
        background: rgba(255, 0, 255, 0.2);
        border: 1px solid #ff00ff;
        margin-left: 20%;
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.3);
    }
    
    .bot-message {
        background: rgba(0, 255, 255, 0.2);
        border: 1px solid #00ffff;
        margin-right: 20%;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(45deg, #00ffff, #ff00ff, #ffff00);
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.8);
        border: 2px solid #00ffff;
        border-radius: 10px;
        color: #00ffff;
        font-family: 'Courier Prime', monospace;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ff00ff;
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.5);
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(0, 0, 0, 0.8);
        border: 2px solid #00ffff;
        border-radius: 10px;
        color: #00ffff;
        font-family: 'Courier Prime', monospace;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #ff00ff;
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.5);
    }
    
    /* Metrics styling */
    .metric-card {
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00ffff;
        font-family: 'Orbitron', monospace;
    }
    
    .metric-label {
        color: #ffffff;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Loading animation */
    .loading-text {
        font-family: 'Orbitron', monospace;
        color: #00ffff;
        text-align: center;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .glitch {
        position: relative;
        color: #00ffff;
        font-size: 1.2rem;
        font-weight: bold;
        text-transform: uppercase;
        animation: glitch 2s infinite;
        text-align: center;
    }
    
    @keyframes glitch {
        0%, 100% { transform: translate(0); }
        20% { transform: translate(-2px, 2px); }
        40% { transform: translate(-2px, -2px); }
        60% { transform: translate(2px, 2px); }
        80% { transform: translate(2px, -2px); }
    }
    
    /* Code blocks */
    .calculation-box {
        background: rgba(0, 255, 0, 0.1);
        border: 1px solid #00ff00;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Courier Prime', monospace;
        font-size: 0.9rem;
    }
    
    .citation-box {
        background: rgba(255, 255, 0, 0.1);
        border: 1px solid #ffff00;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Courier Prime', monospace;
        font-size: 0.9rem;
    }
    
    /* Expandable sections */
    .stExpander > div > div {
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid #00ffff;
        border-radius: 10px;
    }
    
    .stExpander > div > div > div {
        color: #00ffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = None
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'welcome'
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def initialize_ai_system():
    """Initialize the AI system"""
    try:
        # Check for API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            st.stop()
        
        if not st.session_state.system_initialized:
            st.session_state.ai_system = TechnicalDocumentAI(openai_api_key=openai_key)
            st.session_state.system_initialized = True
            
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        st.stop()

# Glitch text effect
def get_glitch_text():
    glitch_texts = [
        'NEURAL_PROCESSING', 'QUANTUM_ANALYSIS', 'DATA_SYNTHESIS', 'AI_INTERFACING',
        'MATRIX_LOADING', 'DOCUMENT_PARSING', 'VECTOR_EMBEDDING', 'KNOWLEDGE_GRAPH'
    ]
    return random.choice(glitch_texts)

# Welcome page
def welcome_page():
    st.markdown('<div class="cyber-title">NEURAL DOCS</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-subtitle">TECHNICAL DOCUMENT AI MATRIX</div>', unsafe_allow_html=True)
    
    # Glitch text effect
    glitch_placeholder = st.empty()
    glitch_placeholder.markdown(f'<div class="glitch">{get_glitch_text()}</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="neon-box">
        <h3 style="color: #00ffff; text-align: center; font-family: 'Orbitron', monospace;">
            🧠 ADVANCED AI DOCUMENT PROCESSING 🧠
        </h3>
        <p style="color: #ffffff; text-align: center; font-size: 1.1rem;">
            Upload technical documents and engage with intelligent AI for complex analysis
        </p>
        <div style="text-align: center; margin-top: 20px;">
            <span style="color: #00ff00;">◉ PDF PARSING</span> &nbsp;&nbsp;&nbsp;
            <span style="color: #ff00ff;">◉ FORMULA EXTRACTION</span> &nbsp;&nbsp;&nbsp;
            <span style="color: #ffff00;">◉ GRAPH RELATIONSHIPS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Capabilities section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🕸️</div>
            <div class="metric-label">Graph + Vector Search</div>
            <div style="color: #888; font-size: 0.8rem; margin-top: 5px;">
                NetworkX + ChromaDB + OpenAI
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🧮</div>
            <div class="metric-label">Math Engine</div>
            <div style="color: #888; font-size: 0.8rem; margin-top: 5px;">
                SymPy + Pint calculations
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🤖</div>
            <div class="metric-label">GPT-4 Reasoning</div>
            <div style="color: #888; font-size: 0.8rem; margin-top: 5px;">
                Natural language processing
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("""
    <div class="neon-box">
        <h4 style="color: #ff00ff; text-align: center; font-family: 'Orbitron', monospace;">
            ⚡ UPLOAD TO NEURAL MATRIX ⚡
        </h4>
        <p style="color: #00ffff; text-align: center;">
            Initialize document processing: PDF files containing technical specifications
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "INITIATE PDF UPLOAD SEQUENCE",
        type=['pdf'],
        key="file_uploader"
    )
    
    if uploaded_file:
        st.session_state.uploaded_files = [uploaded_file]
        
        st.markdown('<div class="neon-box-green">', unsafe_allow_html=True)
        st.markdown("""
        <h4 style="color: #00ff00; text-align: center; font-family: 'Orbitron', monospace;">
            📊 DOCUMENT LOADED 📊
        </h4>
        """, unsafe_allow_html=True)
        
        # Display uploaded file
        file_size = len(uploaded_file.getvalue()) if hasattr(uploaded_file, 'getvalue') else 0
        file_size_str = f"{file_size / (1024*1024):.1f} MB" if file_size > 0 else "Unknown size"
        
        st.markdown(f"""
        <div style="
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid #00ff00;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            font-family: 'Courier Prime', monospace;
        ">
            <strong style="color: #00ff00;">[01] {uploaded_file.name}</strong><br>
            <small style="color: #ffffff; opacity: 0.8;">{file_size_str} • PDF READY FOR NEURAL PROCESSING</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("⚡ ACTIVATE NEURAL PROCESSING ⚡", key="process_btn"):
            st.session_state.current_view = 'processing'
            st.rerun()

# Processing page
def processing_page():
    st.markdown('<div class="cyber-title">NEURAL PROCESSING</div>', unsafe_allow_html=True)
    
    processing_steps = [
        'INITIALIZING AI SYSTEMS...',
        'LOADING PDF STRUCTURE...',
        'EXTRACTING TEXT CONTENT...',
        'PARSING DOCUMENT SECTIONS...',
        'EXTRACTING MATHEMATICAL FORMULAS...',
        'BUILDING VECTOR EMBEDDINGS...',
        'CREATING NEURAL KNOWLEDGE GRAPH...',
        'OPTIMIZING SEARCH INDICES...',
        'CALIBRATING AI RESPONSES...',
        'FINALIZING NEURAL MATRIX...'
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Actual processing
    initialize_ai_system()
    
    uploaded_file = st.session_state.uploaded_files[0]
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process with visual feedback
        for i, step in enumerate(processing_steps):
            progress = (i + 1) / len(processing_steps)
            progress_bar.progress(progress)
            status_text.markdown(f'<div class="loading-text">{step}</div>', unsafe_allow_html=True)
            
            if i == 5:  # At the embedding step, actually process the document
                doc_id = st.session_state.ai_system.ingest_document(tmp_path, uploaded_file.name)
                doc_info = st.session_state.ai_system.get_document_info(doc_id)
                st.session_state.documents[doc_id] = doc_info
            
            time.sleep(0.3)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Mark processing as complete
        st.session_state.processing_complete = True
        
        # Initialize chat with welcome message
        doc_info = list(st.session_state.documents.values())[0]
        welcome_msg = f""">>> NEURAL INTERFACE ACTIVATED <<<

Document matrix successfully integrated: {uploaded_file.name}

◉ SECTIONS: {doc_info.get('sections', 0)}
◉ FORMULAS: {doc_info.get('formulas', 0)}  
◉ TABLES: {doc_info.get('tables', 0)}
◉ PAGES: {doc_info.get('total_pages', 0)}

AI systems online with NEURAL KNOWLEDGE GRAPH. I can now execute complex queries:

• ANALYZE > Extract technical insights with relationship mapping
• SEARCH > Locate requirements using semantic + graph search
• CALCULATE > Perform mathematical computations with dependency tracking
• EXPLAIN > Break down concepts with contextual relationships
• COMPARE > Cross-reference standards with graph connections

Neural processing complete. Knowledge graph activated. Initiate query sequence..."""
        
        st.session_state.chat_messages = [{'role': 'assistant', 'content': welcome_msg}]
        st.session_state.current_view = 'chat'
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.session_state.current_view = 'welcome'
        time.sleep(2)
        st.rerun()

# Chat page
def chat_page():
    # Header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        doc_info = list(st.session_state.documents.values())[0] if st.session_state.documents else {}
        st.markdown(f"""
        <div class="neon-box">
            <h2 style="color: #00ffff; font-family: 'Orbitron', monospace; margin: 0;">
                🤖 NEURAL TECHNICAL AI
            </h2>
            <p style="color: #ffffff; font-family: 'Courier Prime', monospace; margin: 5px 0 0 0;">
                DOCUMENT: {doc_info.get('name', 'Unknown')} | SECTIONS: {doc_info.get('sections', 0)} | FORMULAS: {doc_info.get('formulas', 0)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 RESET MATRIX", key="reset_btn"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Example queries
    st.markdown("""
    <div class="neon-box-purple">
        <h4 style="color: #ff00ff; font-family: 'Orbitron', monospace; margin-bottom: 15px;">
            💡 SAMPLE NEURAL QUERIES
        </h4>
        <div style="font-family: 'Courier Prime', monospace; font-size: 0.9rem;">
            <div style="color: #00ffff; margin: 5px 0;">• "How much ventilation is required for a 6-car carpark?"</div>
            <div style="color: #00ffff; margin: 5px 0;">• "Calculate the minimum ceiling height for office spaces"</div>
            <div style="color: #00ffff; margin: 5px 0;">• "What are the fire resistance requirements for a 20-story building?"</div>
            <div style="color: #00ffff; margin: 5px 0;">• "Compare Class A and Class B ventilation standards"</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_messages:
            if message['role'] == 'assistant':
                # Check if it's a complex AI response with multiple fields
                if isinstance(message['content'], dict):
                    display_ai_response(message['content'])
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong style="color: #00ffff;">🤖 NEURAL AI:</strong><br>
                        <pre style="margin: 10px 0 0 0; white-space: pre-wrap; color: #ffffff;">{message['content']}</pre>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong style="color: #ff00ff;">👤 USER:</strong><br>
                    <pre style="margin: 10px 0 0 0; white-space: pre-wrap; color: #ffffff;">{message['content']}</pre>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_area(
            "Input neural query...",
            key="chat_input",
            placeholder="Ask about your technical document...",
            height=80
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        send_button = st.button("🚀 SEND", key="send_btn")
    
    if send_button and user_input.strip():
        # Add user message
        st.session_state.chat_messages.append({'role': 'user', 'content': user_input})
        
        # Generate AI response
        with st.spinner("🧠 Neural processing..."):
            try:
                bot_response = st.session_state.ai_system.answer_question(user_input)
                st.session_state.chat_messages.append({'role': 'assistant', 'content': bot_response})
            except Exception as e:
                error_response = f">>> NEURAL ERROR <<<\n\nProcessing failed: {str(e)}\n\nReinitializing neural pathways..."
                st.session_state.chat_messages.append({'role': 'assistant', 'content': error_response})
        
        st.rerun()

def display_ai_response(response):
    """Display a complex AI response with proper formatting"""
    
    # Main answer
    st.markdown(f"""
    <div class="chat-message bot-message">
        <strong style="color: #00ffff;">🤖 NEURAL AI:</strong><br>
        <pre style="margin: 10px 0 0 0; white-space: pre-wrap; color: #ffffff;">{response.get('answer', 'No response generated.')}</pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional information in expandable sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculations
        if response.get('calculations'):
            st.markdown('<div class="calculation-box">', unsafe_allow_html=True)
            st.markdown("**🧮 NEURAL CALCULATIONS**")
            for calc in response['calculations']:
                expression = calc.get('expression', '')
                result = calc.get('numeric_result', '')
                st.code(f"{expression} = {result}")
                
                if calc.get('steps'):
                    with st.expander("📊 Calculation Steps"):
                        for step in calc['steps']:
                            st.write(f"• {step}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Supporting evidence
        if response.get('supporting_evidence'):
            with st.expander("📚 Supporting Evidence"):
                for evidence in response['supporting_evidence'][:3]:
                    st.write(f"**Section {evidence.get('section_number', 'Unknown')}:** {evidence.get('section_title', 'N/A')}")
                    st.write(f"Page {evidence.get('page_number', 'N/A')}")
                    st.write(f"{evidence.get('content', 'N/A')[:200]}...")
                    st.write("---")
    
    with col2:
        # Citations
        if response.get('citations'):
            st.markdown('<div class="citation-box">', unsafe_allow_html=True)
            st.markdown("**📖 NEURAL CITATIONS**")
            for citation in response['citations']:
                section = citation.get('section', '')
                title = citation.get('title', '')
                page = citation.get('page', '')
                st.write(f"**Section {section}:** {title}")
                if page:
                    st.write(f"*Page {page}*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Confidence and reasoning
        confidence = response.get('confidence', 0)
        if confidence > 0:
            st.markdown("**📊 NEURAL ANALYSIS**")
            st.progress(confidence, text=f"Confidence: {confidence:.0%}")
        
        if response.get('reasoning_steps'):
            with st.expander("🔍 Reasoning Process"):
                for i, step in enumerate(response['reasoning_steps'], 1):
                    st.write(f"{i}. {step}")

# Main app
def main():
    load_cyberpunk_css()
    initialize_session_state()
    
    # Route to appropriate page
    if st.session_state.current_view == 'welcome':
        welcome_page()
    elif st.session_state.current_view == 'processing':
        processing_page()
    elif st.session_state.current_view == 'chat':
        chat_page()

if __name__ == "__main__":
    main()