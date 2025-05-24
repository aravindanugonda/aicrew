"""
AICrew Research Agent Application

A Streamlit application that uses CrewAI to orchestrate multiple AI agents
for comprehensive research tasks. Supports various search providers and
provides real-time agent interaction monitoring.

"""

import os
import sys
import json
import io
import re
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  

import streamlit as st
import requests
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from dotenv import load_dotenv
import warnings

# Load environment variables
load_dotenv()

class SearchProvider(Enum):
    """Enumeration of available search providers."""
    NO_SEARCH = "No Search Tool (Use LLM Knowledge Only)"
    GOOGLE_SEARCH = "Google Search"


class ModelConfig(Enum):
    """Available Gemini models."""
    GEMINI_1_5_FLASH = "gemini/gemini-1.5-flash"
    GEMINI_2_0_FLASH = "gemini/gemini-2.0-flash"


@dataclass
class AppConfig:
    """Application configuration container."""
    gemini_api_key: str = ""
    google_search_api_key: str = ""
    google_search_cx: str = ""
    search_provider: SearchProvider = SearchProvider.NO_SEARCH
    temperature: float = 0.7
    model: ModelConfig = ModelConfig.GEMINI_1_5_FLASH
    show_agent_details: bool = False
    
    def is_valid_for_research(self) -> tuple[bool, str]:
        """
        Validate configuration for research execution.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.gemini_api_key:
            return False, "Gemini API key is required"
        
        if self.search_provider == SearchProvider.GOOGLE_SEARCH:
            if not self.google_search_api_key:
                return False, "Google Search API key is required for Google Search"
            if not self.google_search_cx:
                return False, "Google Custom Search Engine ID (CX) is required"
        
        return True, ""


class GoogleSearchInput(BaseModel):
    """Input schema for Google Search Tool."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str = Field(description="The search query to perform")


class GoogleSearchTool(BaseTool):
    """
    Google Custom Search API tool for CrewAI agents.
    
    Provides web search capabilities using Google's Custom Search API
    with comprehensive error handling and response formatting.
    """
    name: str = "google_search"
    description: str = (
        "Search the web for information using Google Custom Search API. "
        "Returns comprehensive results with titles, URLs, and descriptions."
    )
    args_schema: Type[BaseModel] = GoogleSearchInput
    _api_key: str = PrivateAttr()
    _cx: str = PrivateAttr()

    def __init__(self, api_key: str, cx: str, **kwargs):
        """
        Initialize Google Search Tool.
        
        Args:
            api_key: Google Cloud API key with Custom Search API enabled
            cx: Google Custom Search Engine ID
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._cx = cx

    def _run(self, query: str) -> str:
        """
        Execute Google Custom Search and return formatted results.
        
        Args:
            query: Search query string
            
        Returns:
            Formatted search results or error message
        """
        if not self._api_key:
            return "Error: Google Search API Key is not configured."
        if not self._cx:
            return "Error: Google Custom Search Engine ID (CX) is not configured."
        
        response: Optional[requests.Response] = None
        try:
            response = self._make_search_request(query)
            return self._format_search_results(response.json())
            
        except requests.exceptions.HTTPError as e:
            return self._handle_http_error(e, response)
        except requests.exceptions.RequestException as e:
            return f"Network error during search: {str(e)}"
        except Exception as e:
            return f"Unexpected error during search: {str(e)}"

    def _make_search_request(self, query: str) -> requests.Response:
        """Make HTTP request to Google Custom Search API."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self._api_key,
            "cx": self._cx,
            "q": query
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response

    def _format_search_results(self, results: Dict[str, Any]) -> str:
        """Format search results into readable text."""
        formatted_text = "Google Search Results:\n\n"
        
        if "items" not in results:
            formatted_text += "No results found.\n"
            if "error" in results:
                formatted_text += f"API Error: {results['error'].get('message', 'Unknown error')}\n"
            return formatted_text

        for i, item in enumerate(results["items"], 1):
            title = item.get("title", "No title")
            link = item.get("link", "No link")
            snippet = item.get("snippet", "No description")

            formatted_text += f"{i}. {title}\n"
            formatted_text += f"   URL: {link}\n"
            formatted_text += f"   Description: {snippet}\n\n"

        return formatted_text

    def _handle_http_error(self, error: requests.exceptions.HTTPError, response: Optional[requests.Response]) -> str:
        """Handle HTTP errors with detailed error messages."""
        if response is None:
            return f"HTTP Error: {str(error)} (No response available)"
            
        try:
            error_content = response.json()
            error_message = error_content.get("error", {}).get("message", "Unknown error")
        except (json.JSONDecodeError, AttributeError):
            error_message = response.text
        
        return f"Google Search API Error (HTTP {response.status_code}): {error_message}"


class AgentLogger:
    """
    Manages logging and display of agent interactions in Streamlit UI.
    
    Provides real-time updates of agent outputs with organized display
    using expandable sections for each agent type.
    """
    
    def __init__(self, container: Any, show_details: bool = True): 
        """
        Initialize agent logger.
        
        Args:
            container: Streamlit container for displaying logs
            show_details: Whether to show detailed agent interactions
        """
        self.container = container
        self.show_details = show_details
        self.agents_data: Dict[str, List[Dict[str, str]]] = {}
        self.expanders: Dict[str, Any] = {} 
        
        if self.show_details:
            self._initialize_ui()
    
    def _initialize_ui(self):
        """Initialize Streamlit UI components for agent logging."""
        with self.container:
            st.markdown("## Agent Outputs")
            agent_types = ["Research Specialist", "Information Analyst", "Content Writer"]
            
            for agent_type in agent_types:
                self.expanders[agent_type] = st.expander(
                    agent_type, 
                    expanded=(agent_type == "Research Specialist")
                )
    
    def log_output(self, agent_role: str, output_text: str):
        """
        Log agent output and update UI display.
        
        Args:
            agent_role: Role/type of the agent
            output_text: Output text from the agent
        """
        if not self.show_details or len(output_text.strip()) == 0:
            return
            
        # Initialize agent data if not exists
        if agent_role not in self.agents_data:
            self.agents_data[agent_role] = []
        
        # Add new output
        self.agents_data[agent_role].append({"content": output_text})
        self._update_display(agent_role)
    
    def _update_display(self, agent_role: str):
        """Update the display for a specific agent."""
        expander = self._find_expander(agent_role)
        if not expander:
            return
        
        # Build markdown content from all outputs
        content = ""
        for item in self.agents_data[agent_role]:
            content += f"```\n{item['content']}\n```\n\n"
        
        expander.markdown(content)
    
    def _find_expander(self, agent_role: str) -> Optional[Any]:
        """Find the appropriate expander for an agent role."""
        for key, expander in self.expanders.items():
            if key.lower() in agent_role.lower():
                return expander
        return None


class OutputCapture:
    """
    Captures and processes stdout/stderr for agent interaction monitoring.
    
    Redirects system output to extract meaningful agent communications
    and displays them in the Streamlit interface.
    """
    
    def __init__(self, agent_logger: Optional[AgentLogger] = None):
        """
        Initialize output capture.
        
        Args:
            agent_logger: Optional logger for displaying captured output
        """
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.agent_logger = agent_logger
        self.buffer = io.StringIO()
        self.current_agent: Optional[str] = None
        self.seen_messages: set = set()
        self.accumulating_buffer = ""
        
        # Patterns for identifying agent types and extracting answers
        self.agent_patterns = {
            "research": re.compile(r".*Research Specialist.*", re.IGNORECASE),
            "analyst": re.compile(r".*Information Analyst.*", re.IGNORECASE),
            "writer": re.compile(r".*Content Writer.*", re.IGNORECASE)
        }
        self.final_answer_pattern = re.compile(r"## Final Answer:\s*(.*)", re.DOTALL)
        
    def write(self, message: str):
        """
        Process and capture output messages.
        
        Args:
            message: Output message to process
        """
        # Always write to terminal
        self.terminal_stdout.write(message)
        self.buffer.write(message)
        
        if not self.agent_logger or not message.strip():
            return
            
        # Clean message and update buffer
        clean_message = self._clean_message(message)
        self.accumulating_buffer += clean_message + "\n"
        
        # Update current agent if pattern matches
        self._update_current_agent(message)
        
        # Extract and log final answers
        if self.current_agent:
            self._extract_and_log_answer()
            
        # Prevent buffer from growing too large
        self._trim_buffer_if_needed()
    
    def _clean_message(self, message: str) -> str:
        """Remove ANSI color codes and terminal artifacts."""
        return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', message).strip()
    
    def _update_current_agent(self, message: str):
        """Update current agent based on message patterns."""
        for agent_type, pattern in self.agent_patterns.items():
            if pattern.search(message):
                agent_mapping = {
                    "research": "Research Specialist",
                    "analyst": "Information Analyst", 
                    "writer": "Content Writer"
                }
                self.current_agent = agent_mapping[agent_type]
                break
    
    def _extract_and_log_answer(self):
        """Extract final answers and log them."""
        # Check if agent_logger and current_agent are not None before calling
        if self.agent_logger is None or self.current_agent is None:
            return
            
        answer_match = self.final_answer_pattern.search(self.accumulating_buffer)
        if answer_match:
            answer = answer_match.group(0).strip()
            if answer not in self.seen_messages:
                self.agent_logger.log_output(self.current_agent, answer)
                self.seen_messages.add(answer)
                self.accumulating_buffer = ""
    
    def _trim_buffer_if_needed(self):
        """Trim buffer to prevent memory issues."""
        if len(self.accumulating_buffer) > 50000:
            self.accumulating_buffer = self.accumulating_buffer[-20000:]
    
    def flush(self):
        """Flush output buffer."""
        self.terminal_stdout.flush()
    
    def get_output(self) -> str:
        """Get captured output."""
        return self.buffer.getvalue()
    
    def reset(self):
        """Reset capture state."""
        self.buffer = io.StringIO()
        self.current_agent = None
        self.seen_messages = set()
        self.accumulating_buffer = ""


class ResearchEngine:
    """
    Core research engine that orchestrates CrewAI agents for research tasks.
    
    Manages the creation of agents, tasks, and crew coordination for
    comprehensive research with optional search tool integration.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize research engine with configuration.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.tools: List[BaseTool] = []
        
    def setup_tools(self):
        """Setup search tools based on configuration."""
        self.tools = []
        
        if self.config.search_provider == SearchProvider.GOOGLE_SEARCH:
            search_tool = GoogleSearchTool(
                api_key=self.config.google_search_api_key,
                cx=self.config.google_search_cx
            )
            self.tools.append(search_tool)
    
    def create_llm(self) -> LLM:
        """
        Create and configure Gemini LLM instance.
        
        Returns:
            Configured LLM instance
        """
        return LLM(
            model=self.config.model.value,
            api_key=self.config.gemini_api_key,
            temperature=self.config.temperature,
        )
    
    def create_agents(self, topic: str, llm: LLM) -> tuple[Agent, Agent, Agent]:
        """
        Create specialized research agents.
        
        Args:
            topic: Research topic
            llm: Language model instance
            
        Returns:
            Tuple of (researcher, analyst, writer) agents
        """
        researcher = Agent(
            role="Research Specialist",
            goal=f"Research {topic} thoroughly and provide comprehensive information",
            backstory="You are an expert researcher with a talent for finding detailed information on any subject.",
            verbose=True,
            llm=llm,
            tools=self.tools
        )
        
        analyst = Agent(
            role="Information Analyst",
            goal=f"Analyze research findings on {topic} and extract key insights",
            backstory="You are a skilled analyst with expertise in synthesizing information and identifying patterns.",
            verbose=True,
            llm=llm,
            tools=self.tools
        )
        
        writer = Agent(
            role="Content Writer",
            goal=f"Create a well-structured, informative report on {topic}",
            backstory="You are a talented writer with a knack for clarity and engaging content.",
            verbose=True,
            llm=llm
        )
        
        return researcher, analyst, writer
    
    def create_tasks(self, topic: str, researcher: Agent, analyst: Agent, writer: Agent) -> List[Task]:
        """
        Create research tasks with proper dependencies.
        
        Args:
            topic: Research topic
            researcher: Research agent
            analyst: Analysis agent
            writer: Writing agent
            
        Returns:
            List of configured tasks
        """
        # Determine search instructions based on provider
        search_instruction = ""
        if self.config.search_provider != SearchProvider.NO_SEARCH:
            search_instruction = (
                "Use the search tool to gather comprehensive information including "
                "recent developments, key concepts, historical context, and relevant "
                "statistics. Verify information from multiple sources when possible."
            )
        
        research_task = Task(
            description=f"Research the topic: {topic}. {search_instruction}",
            agent=researcher,
            expected_output="Detailed research findings with all relevant information and sources."
        )
        
        verify_instruction = ""
        if self.config.search_provider != SearchProvider.NO_SEARCH:
            verify_instruction = "Use the search tool to verify or expand on information as needed."
            
        analysis_task = Task(
            description=(
                f"Analyze the research findings on {topic}. Identify key trends, "
                f"patterns, insights, and implications. {verify_instruction}"
            ),
            agent=analyst,
            expected_output="In-depth analysis with key insights, trends, and interpretation of the research findings.",
            context=[research_task]
        )
        
        writing_task = Task(
            description=(
                f"Using the research and analysis, create a comprehensive report on {topic}. "
                "The report should be well-structured, informative, and accessible to a general audience."
            ),
            agent=writer,
            expected_output="A complete, well-structured report on the topic with all key information presented clearly.",
            context=[analysis_task]
        )
        
        return [research_task, analysis_task, writing_task]
    
    def execute_research(self, topic: str, agent_logger: Optional[AgentLogger] = None) -> str:
        """
        Execute research workflow using CrewAI.
        
        Args:
            topic: Research topic
            agent_logger: Optional logger for UI updates
            
        Returns:
            Research results as string
            
        Raises:
            Exception: If research execution fails
        """
        # Setup output capture
        stdout_capture = OutputCapture(agent_logger=agent_logger)
        original_stdout = sys.stdout
        
        try:
            sys.stdout = stdout_capture
            
            # Initialize components
            self.setup_tools()
            llm = self.create_llm()
            researcher, analyst, writer = self.create_agents(topic, llm)
            tasks = self.create_tasks(topic, researcher, analyst, writer)
            
            # Create and run crew
            crew = Crew(
                agents=[researcher, analyst, writer],
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            
            # Extract result content
            if hasattr(result, 'raw'):
                return result.raw
            else:
                return str(result)
                
        finally:
            sys.stdout = original_stdout


def initialize_session_state():
    """Initialize Streamlit session state with default values."""
    defaults = {
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
        "google_search_api_key": os.environ.get("GOOGLE_SEARCH_API_KEY", ""),
        "google_search_cx": os.environ.get("GOOGLE_SEARCH_CX", "")
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def setup_page():
    """Configure Streamlit page settings and display header."""
    st.set_page_config(
        page_title="AICrew Research Agent",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AICrew Research Agent")
    st.markdown("Research any topic using multiple AI agents powered by Gemini")


def create_sidebar_config() -> AppConfig:
    """
    Create sidebar configuration interface and return AppConfig.
    
    Returns:
        Configured AppConfig object
    """
    with st.sidebar:
        st.header("Configuration")
        
        # Search provider selection
        search_provider_str = st.selectbox(
            "Search Provider",
            options=[provider.value for provider in SearchProvider],
            index=0,
            help="Select which search API to use for research, or use no search tool"
        )
        search_provider = SearchProvider(search_provider_str)
        
        # API key inputs
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Enter your Gemini API key from Google AI Studio"
        )
        st.session_state.gemini_api_key = gemini_api_key
        
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
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.1,
                help="Higher values make output more creative, lower values more deterministic"
            )
            
            model_str = st.selectbox(
                "Gemini Model",
                options=[model.value for model in ModelConfig],
                index=0,
                help="Select which Gemini model to use"
            )
            model = ModelConfig(model_str)
            
            show_agent_details = st.checkbox(
                "Show detailed agent interactions",
                value=False, 
                help="Display detailed input/output for each agent during the research process"
            )
        
        # About section
        st.markdown("### About")
        st.markdown("""
        This application uses:
        - **CrewAI**: To orchestrate multiple AI agents
        - **Google Gemini**: For AI language capabilities  
        - **Search Options**:
          - No Search Tool (relies on LLM knowledge)
          - Google Search for web search (requires Google Search API Key and CX ID)
        - **Streamlit**: For the user interface

        No data is stored persistently - all processing happens in your local session.
        """)
    
    return AppConfig(
        gemini_api_key=gemini_api_key or "",
        google_search_api_key=google_search_api_key or "",
        google_search_cx=google_search_cx or "",
        search_provider=search_provider,
        temperature=temperature,
        model=model,
        show_agent_details=show_agent_details
    )


def create_research_interface() -> tuple[str, Optional[str]]:
    """
    Create research input interface.
    
    Returns:
        Tuple of (research_topic, research_focus)
    """
    research_topic = st.text_input(
        "Research Topic",
        placeholder="Enter a topic to research (e.g., Quantum Computing, Climate Change)",
        help="Be specific for better results"
    )
    
    research_focus = None
    with st.expander("Research Focus (Optional)"):
        research_focus = st.text_area(
            "Specific aspects to focus on",
            placeholder="Enter any specific aspects you want the research to focus on...",
            help="Leave blank for a general overview"
        )
        if not research_focus.strip():
            research_focus = None
    
    return research_topic, research_focus


def execute_research_workflow(config: AppConfig, topic: str, focus: Optional[str] = None) -> str:
    """
    Execute the complete research workflow.
    
    Args:
        config: Application configuration
        topic: Research topic
        focus: Optional research focus
        
    Returns:
        Research results as string
    """
    # Validate configuration
    is_valid, error_message = config.is_valid_for_research()
    if not is_valid:
        return f"ERROR: {error_message}"
    
    if not topic.strip():
        return "ERROR: Please enter a research topic."
    
    # Build full topic including focus
    full_topic = f"{topic}{': ' + focus if focus else ''}"
    
    # Initialize UI components
    progress_placeholder = st.empty()
    agent_logs_container = st.container()
    
    try:
        # Setup progress tracking
        progress_placeholder.info("Initializing research agents...")
        
        # Initialize agent logger
        agent_logger = AgentLogger(agent_logs_container, config.show_agent_details)
        
        # Create and execute research
        engine = ResearchEngine(config)
        
        progress_placeholder.info("Assembling AI research crew...")
        progress_placeholder.warning("Research in progress... This may take several minutes.")
        
        result = engine.execute_research(full_topic, agent_logger)
        
        progress_placeholder.success("Research completed successfully!")
        return result
        
    except Exception as e:
        progress_placeholder.error("Research failed!")
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
    
    # Create research interface
    research_topic, research_focus = create_research_interface()
    
    # Research execution
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        start_research = st.button("Start Research", type="primary", use_container_width=True)
    
    # Execute research when button clicked
    if start_research:
        with st.container():
            with st.spinner("Running research..."):
                research_result = execute_research_workflow(
                    config=config,
                    topic=research_topic,
                    focus=research_focus
                )
                
                st.markdown("## Research Results")
                if research_result and not research_result.startswith("ERROR:"):
                    st.markdown(research_result)
                else:
                    st.error(research_result or "No results or an error occurred. Please check your configuration and try again.")


if __name__ == "__main__":
    main()