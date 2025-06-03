# Use pysqlite3 as sqlite3 before any other imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from crew_orchestrator import (
    AppConfig,
    SearchProvider,
    ResearchCrew,
    StockAnalysisCrew
)
from dotenv import load_dotenv

# Define crew type strings
RESEARCH_CREW = "Research Crew"
STOCK_ANALYSIS_CREW = "Stock Analysis Crew"

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize Streamlit session state with default values."""
    defaults = {
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "google_search_api_key": os.environ.get("GOOGLE_SEARCH_API_KEY", ""),
        "google_search_cx": os.environ.get("GOOGLE_SEARCH_CX", "")
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def setup_page():
    """Configure Streamlit page settings and display header."""
    st.set_page_config(
        page_title="AICrew Orchestrator",
        page_icon="🤖",
        layout="wide"
    )
    # Inject custom CSS for more readable monospace font
    st.markdown(
        """
        <style>
        code, pre, .stCode, .stMarkdown code {
            font-family: 'Fira Mono', 'JetBrains Mono', 'Source Code Pro', 'Menlo', 'Consolas', 'Monaco', 'monospace' !important;
            font-size: 1.05em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("🤖 AICrew Orchestrator")
    st.markdown("Orchestrate multiple AI agents for different tasks")

def create_sidebar_config() -> AppConfig:
    """
    Create sidebar configuration interface and return AppConfig.
    
    Returns:
        Configured AppConfig object
    """
    with st.sidebar:
        st.header("Configuration")
        
        # Crew type selection
        crew_type_str = st.selectbox(
            "Crew Type",
            options=[
                RESEARCH_CREW, 
                STOCK_ANALYSIS_CREW
            ],
            index=0,
            help="Select which crew to use for the task"
        )
        crew_type = crew_type_str
        
        # Search provider selection
        search_provider_str = st.selectbox(
            "Search Provider",
            options=[provider.value for provider in SearchProvider],
            index=0,
            help="Select which search API to use (if needed)"
        )
        search_provider = SearchProvider(search_provider_str)
        
        # API key inputs
        model_provider = st.selectbox(
            "Model Provider",
            options=["Gemini", "OpenAI"],
            index=0,
            help="Select which model provider to use"
        )
        # Initialize API key variables
        gemini_api_key = st.session_state.gemini_api_key
        openai_api_key = st.session_state.openai_api_key
        if model_provider == "Gemini":
            gemini_api_key = st.text_input(
                "Gemini API Key",
                value=gemini_api_key,
                type="password",
                help="Enter your Gemini API key from Google AI Studio"
            )
            st.session_state.gemini_api_key = gemini_api_key
        elif model_provider == "OpenAI":
            openai_api_key = st.text_input(
                "OpenAI API Key",
                value=openai_api_key,
                type="password",
                help="Enter your OpenAI API key"
            )
            st.session_state.openai_api_key = openai_api_key
        
        google_search_api_key = ""
        google_search_cx = ""
        
        if search_provider == SearchProvider.GOOGLE_SEARCH:
            google_search_api_key = st.text_input(
                "Google Search API Key",
                value=st.session_state.google_search_api_key,
                type="password",
                help="Enter your Google Cloud API key enabled for Custom Search API"
            )
            st.session_state.google_search_api_key = google_search_api_key
            
            google_search_cx = st.text_input(
                "Google Custom Search Engine ID (CX)",
                value=st.session_state.google_search_cx,
                type="password",
                help="Enter your Google Custom Search Engine ID (CX)"
            )
            st.session_state.google_search_cx = google_search_cx
        
        # Advanced options
        with st.expander("Advanced Options"):
            # Provide examples based on selected provider
            if model_provider == "Gemini":
                example = "gemini/gemini-2.0-flash-lite"
            else:  # OpenAI
                example = "gpt-4.1-nano-2025-04-14"
            # Dynamically update model_name default when provider changes
            model_name = st.text_input(
                "Model Name",
                value=example,
                help=f"Enter model name (e.g. '{example}')"
            )
        
        # About section
        st.markdown("### About")
        st.markdown("""
        This application uses:
        - **CrewAI**: To orchestrate multiple AI agents
        - **Google Gemini/OpenAI**: For AI language capabilities
        - **Streamlit**: For the user interface
        """)
    
    return AppConfig(
        gemini_api_key=gemini_api_key or "",
        openai_api_key=openai_api_key or "",
        google_search_api_key=google_search_api_key or "",
        google_search_cx=google_search_cx or "",
        search_provider=search_provider,
        model_provider=model_provider,
        model_name=model_name,
        crew_type=crew_type
    )

def create_task_interface(crew_type: str) -> str:
    """
    Create task input interface based on crew type.
    
    Returns:
        Task description string
    """
    if crew_type == RESEARCH_CREW:
        prompt = "Enter research topic (e.g., Quantum Computing, Climate Change)"
        placeholder = "What would you like to research?"
    elif crew_type == STOCK_ANALYSIS_CREW:
        prompt = "Enter stock symbol or company name (e.g., AAPL, Tesla)"
        placeholder = "Which stock would you like to analyze?"
    else:
        prompt = "Enter your task description"
        placeholder = "Describe your task"
    
    return st.text_input(
        prompt,
        placeholder=placeholder,
        help="Be specific for better results"
    )

def execute_workflow(config: AppConfig, task: str) -> str:
    """
    Execute the crew workflow.
    
    Args:
        config: Application configuration
        task: Task description
        
    Returns:
        Results as string
    """
    if not task.strip():
        return "ERROR: Please enter a task description."
    
    try:
        # Create appropriate crew
        if config.crew_type == RESEARCH_CREW:
            crew = ResearchCrew(config)
        elif config.crew_type == STOCK_ANALYSIS_CREW:
            crew = StockAnalysisCrew(config)
        else:
            return "ERROR: Unsupported crew type."
            
        # Execute crew workflow
        return crew.execute(task)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"ERROR: {str(e)}\n\nDetails:\n{error_details}"

def main():
    """Main application entry point."""
    # Initialize application
    setup_page()
    initialize_session_state()
    
    # Create configuration interface
    config = create_sidebar_config()
    
    # Create task interface
    task = create_task_interface(config.crew_type)
    
    # Execution
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        start_task = st.button("Start Task", type="primary", use_container_width=True)
    
    # Execute when button clicked
    if start_task:
        with st.container():
            with st.spinner("Running task..."):
                result = execute_workflow(config, task)
                
                st.markdown("## Task Results")
                if result and not result.startswith("ERROR:"):
                    st.markdown(result, unsafe_allow_html=True)
                else:
                    st.error(result or "No results or an error occurred. Please check your configuration and try again.")

if __name__ == "__main__":
    main()