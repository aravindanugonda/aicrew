import os
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import json
import io
from typing import Type, Dict, Any, List, Optional, Callable, Union
import streamlit as st
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from dotenv import load_dotenv
import traceback
import warnings
import requests
import threading
import time
import re

# Suppress Pydantic warnings about callback functions
warnings.filterwarnings("ignore", message=".*is not a Python type.*")

# Load environment variables from .env file (for local development)
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AICrew Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("🔍 AICrew Research Assistant")
st.markdown("Research any topic using multiple AI agents powered by Gemini")

# Google Search Tool implementation
class GoogleSearchInput(BaseModel):
    """Input schema for Google Search Tool."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str = Field(description="The search query to perform")

class GoogleSearchTool(BaseTool):
    name: str = "google_search"
    description: str = "Search the web for information using Google Custom Search API and return comprehensive results."
    args_schema: Type[BaseModel] = GoogleSearchInput
    _api_key: str = PrivateAttr()
    _cx: str = PrivateAttr()

    def __init__(self, api_key: str, cx: str, **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._cx = cx

    def _run(self, query: str) -> str:
        """Execute Google Custom Search and return results."""
        if not self._api_key:
            return "Error: Google Search API Key is not configured for GoogleSearchTool."
        if not self._cx:
            return "Error: Google Custom Search Engine ID (CX) is not configured for GoogleSearchTool."
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self._api_key,
                "cx": self._cx,
                "q": query
            }
            response = requests.get(url, params=params)
            # print(f"[DEBUG] Google Search API URL: {response.url}")
            # print(f"[DEBUG] Google Search API status: {response.status_code}")
            # print(f"[DEBUG] Google Search API response: {response.text}")
            response.raise_for_status()
            results = response.json()

            formatted_text = "Google Search Results:\n\n"
            if "items" in results:
                for i, item in enumerate(results["items"], 1):
                    name = item.get("title", "No title")
                    url = item.get("link", "No link")  # fix: should be 'link' not 'url'
                    snippet = item.get("snippet", "No description")

                    formatted_text += f"{i}. {name}\n"
                    formatted_text += f"   URL: {url}\n"
                    formatted_text += f"   Description: {snippet}\n\n"
            else:
                formatted_text += "No results found or error in response structure.\n"
                formatted_text += f"Raw response: {json.dumps(results, indent=2)}\n"

            return formatted_text

        except requests.exceptions.HTTPError as http_err:
            error_content = "Unknown error"
            try:
                error_content = response.json() # type: ignore
            except json.JSONDecodeError:
                error_content = response.text # type: ignore
            print(f"[ERROR] Google Search HTTPError: {str(http_err)} Response: {error_content}")
            return f"Error searching with Google Search (HTTP {response.status_code}): {str(http_err)}\nResponse: {error_content}"
        except Exception as e:
            print(f"[ERROR] Google Search Exception: {str(e)}")
            return f"Error searching with Google Search: {str(e)}"

# Initialize session state for API keys if they don't exist
# On first load, try to get values from environment variables
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
if "google_search_api_key" not in st.session_state: # New state for Google Search API key
    st.session_state.google_search_api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
if "google_search_cx" not in st.session_state:
    st.session_state.google_search_cx = os.environ.get("GOOGLE_SEARCH_CX", "")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    
    # Add search provider selection
    search_provider = st.selectbox(
        "Search Provider",
        options=["No Search Tool (Use LLM Knowledge Only)", "Google Search"],
        index=0,
        help="Select which search API to use for research, or use no search tool"
    )

    # API key inputs using session state
    gemini_api_key_input = st.text_input( # Renamed variable to avoid conflict
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Enter your Gemini API key from Google AI Studio"
    )
    # Store in session state (not in environment variables)
    st.session_state.gemini_api_key = gemini_api_key_input

    # Show the appropriate API key input based on selection
    if search_provider == "Google Search":
        google_search_api_key_input = st.text_input( # New input for Google Search API Key
            "Google Search API Key",
            value=st.session_state.google_search_api_key,
            type="password",
            help="Enter your Google Cloud API key enabled for Custom Search API"
        )
        st.session_state.google_search_api_key = google_search_api_key_input

        google_search_cx_input = st.text_input( # Renamed variable
            "Google Custom Search Engine ID (CX)",
            value=st.session_state.google_search_cx,
            type="password",
            help="Enter your Google Custom Search Engine ID (CX)"
        )
        st.session_state.google_search_cx = google_search_cx_input

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

        gemini_model = st.selectbox(
            "Gemini Model",
            options=["gemini/gemini-1.5-flash", "gemini/gemini-2.0-flash"],
            index=0,
            help="Select which Gemini model to use"
        )
        
        show_agent_details = st.checkbox(
            "Show detailed agent interactions",
            value=False, 
            help="Display detailed input/output for each agent during the research process"
        )

    # Update About section to include info about no search option
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

# Main content area
research_topic = st.text_input(
    "Research Topic",
    placeholder="Enter a topic to research (e.g., Quantum Computing, Climate Change)",
    help="Be specific for better results"
)

# Optional research focus
with st.expander("Research Focus (Optional)"):
    research_focus = st.text_area(
        "Specific aspects to focus on",
        placeholder="Enter any specific aspects you want the research to focus on...",
        help="Leave blank for a general overview"
    )

# Research button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    start_research = st.button("Start Research", type="primary", use_container_width=True)

# Progress indicator and results area
progress_placeholder = st.empty()
agent_logs_container = st.container()
results_container = st.container()

# Custom Agent Logger for UI display
class AgentLogger:
    def __init__(self, container, show_details=True):
        self.container = container
        self.show_details = show_details
        self.agents_data = {}
        self.expanders = {}
        
        # Initialize expanders for each agent
        if self.show_details:
            with self.container:
                st.markdown("## Agent Outputs")
                self.expanders["Research Specialist"] = st.expander("Research Specialist", expanded=True)
                self.expanders["Information Analyst"] = st.expander("Information Analyst", expanded=True)
                self.expanders["Content Writer"] = st.expander("Content Writer", expanded=True)
    
    def log_input(self, agent_role, input_text):
        # Skip logging inputs entirely - we only want to show outputs
        pass
    
    def log_output(self, agent_role, output_text):
        if not self.show_details:
            return
            
        if agent_role not in self.agents_data:
            self.agents_data[agent_role] = []
        
        self.agents_data[agent_role].append({
            "content": output_text
        })
        
        self._update_display(agent_role)
    
    def _update_display(self, agent_role):
        # Find the appropriate expander
        expander = None
        for key, exp in self.expanders.items():
            if key in agent_role:
                expander = exp
                break
        
        if not expander:
            # Default to first expander if no match
            expander = list(self.expanders.values())[0]
            
        # Build the markdown content - only showing outputs
        content = ""
        for item in self.agents_data[agent_role]:
            content += f"```\n{item['content']}\n```\n\n"
        
        # Update the expander content
        expander.markdown(content)

# Class for capturing stdout and stderr to display in Streamlit UI
class OutputCapture:
    def __init__(self, agent_logger=None):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.agent_logger = agent_logger
        self.buffer = io.StringIO()
        self.current_agent = None
        self.seen_messages = set()
        self.final_answer_pattern = re.compile(r"## Final Answer:\s*(.*)", re.DOTALL)
        
        # Agent identification patterns
        self.patterns = {
            "research": re.compile(r".*Research Specialist.*", re.IGNORECASE),
            "analyst": re.compile(r".*Information Analyst.*", re.IGNORECASE),
            "writer": re.compile(r".*Content Writer.*", re.IGNORECASE)
        }
        
        # Buffer for accumulating output until we see a complete pattern
        self.accumulating_buffer = ""
        
    def write(self, message):
        # Write to the original terminal
        self.terminal_stdout.write(message)
        
        # Store in buffer
        self.buffer.write(message)
        
        # If no agent_logger or empty message, nothing to do
        if not self.agent_logger or not message.strip():
            return
            
        # Clean the message (remove ANSI color codes and other terminal artifacts)
        clean_message = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', message).strip()
        
        # Update the accumulating buffer
        self.accumulating_buffer += clean_message + "\n"
        
        # Check if this contains agent identification
        for agent_type, pattern in self.patterns.items():
            if pattern.search(message):
                if agent_type == "research":
                    self.current_agent = "Research Specialist"
                elif agent_type == "analyst":
                    self.current_agent = "Information Analyst"
                elif agent_type == "writer":
                    self.current_agent = "Content Writer"
        
        # If we have identified an agent
        if self.current_agent:
            # Look for final answers - these are the most important outputs
            answer_match = self.final_answer_pattern.search(self.accumulating_buffer)
            if answer_match:
                answer = answer_match.group(0).strip()  # Get the full match with "## Final Answer:"
                if answer not in self.seen_messages:
                    self.agent_logger.log_output(self.current_agent, answer)
                    self.seen_messages.add(answer)
                    # Clear the buffer after capturing the answer
                    self.accumulating_buffer = ""
                    
            # If buffer gets too large, trim it to avoid memory issues
            if len(self.accumulating_buffer) > 50000:
                self.accumulating_buffer = self.accumulating_buffer[-20000:]
    
    def flush(self):
        self.terminal_stdout.flush()
    
    def get_output(self):
        return self.buffer.getvalue()
    
    def reset(self):
        self.buffer = io.StringIO()
        self.current_agent = None
        self.seen_messages = set()
        self.accumulating_buffer = ""

# Define callback functions to monitor agent interactions
def create_step_callback(agent_logger):
    """Create a step callback function that logs agent interactions."""
    
    def step_callback(formatted_answer):
        """Callback function that logs agent steps during execution.
        This function is called by CrewAI with the formatted answer from each agent step.
        """
        # Try to determine which agent is responding based on content of the answer
        agent_role = "Agent"  # Default role
        
        # Look for clues in the formatted answer to identify the agent
        if "research" in formatted_answer.lower() and any(term in formatted_answer.lower() for term in ["found", "searched", "discovered", "information", "sources"]):
            agent_role = "Research Specialist"
        elif "analy" in formatted_answer.lower() and any(term in formatted_answer.lower() for term in ["pattern", "trend", "insight", "data", "findings"]):
            agent_role = "Information Analyst"
        elif any(term in formatted_answer.lower() for term in ["report", "article", "summary", "write", "written", "document"]):
            agent_role = "Content Writer"
        
        # Log the response from the agent with the determined role
        agent_logger.log_output(agent_role, formatted_answer[:5000] + ("..." if len(formatted_answer) > 5000 else ""))
        
        # After a response, log a placeholder for the next input
        # This creates a more conversational view in the UI
        if agent_role == "Research Specialist":
            next_agent = "Information Analyst"
            agent_logger.log_input(next_agent, "Analyzing research findings...")
        elif agent_role == "Information Analyst":
            next_agent = "Content Writer"
            agent_logger.log_input(next_agent, "Preparing to write report based on analysis...")
            
    return step_callback

# Modify function to support LinkUp and no-tool option
def run_crewai_research(topic, focus=None, gemini_api_key=None,
                        google_search_api_key=None, google_search_cx=None, 
                        search_provider="No Search Tool (Use LLM Knowledge Only)",
                        model="gemini/gemini-1.5-flash", temp=0.7, show_details=False):
    """
    Run a research task using CrewAI and return the results
    """
    if not gemini_api_key:
        return "ERROR: Please enter your Gemini API key in the sidebar."
    
    if search_provider == "Google Search" and (not google_search_api_key or not google_search_cx):
        return "ERROR: Please enter your Google Search API Key and Custom Search Engine ID (CX) in the sidebar for Google Search."

    if not topic:
        return "ERROR: Please enter a research topic."
    
    try:
        # Set up progress message
        progress_placeholder.info("Initializing research agents...")
        
        # Initialize agent logger
        agent_logger = AgentLogger(agent_logs_container, show_details=show_details)
        
        # Pre-populate the logger with task descriptions to create the narrative flow
        if show_details:
            agent_logger.log_input(
                "Research Specialist", 
                f"Research task: Thoroughly research the topic '{topic}{': ' + focus if focus else ''}' and provide comprehensive information."
            )
        
        # Initialize output capture to redirect terminal output to UI
        stdout_capture = OutputCapture(agent_logger=agent_logger if show_details else None)
        sys.stdout = stdout_capture
        
        # Initialize Gemini LLM
        gemini_llm = LLM(
            model=model,
            api_key=gemini_api_key,
            temperature=temp,
        )
        
        # Initialize the appropriate search tool based on selection
        tools = []
        if search_provider == "Google Search":
            search_tool = GoogleSearchTool(api_key=google_search_api_key, cx=google_search_cx)
            tools = [search_tool]
            progress_placeholder.info("Initializing Google Search tool...")
        else:
            # No search tool option
            progress_placeholder.info("Running without search tools (using LLM knowledge only)...")
        
        # Include focus in the topic if provided
        full_topic = f"{topic}{': ' + focus if focus else ''}"
        
        # Create researcher agent with verbose enabled
        progress_placeholder.info("Creating research specialist agent...")
        researcher = Agent(
            role="Research Specialist",
            goal=f"Research {full_topic} thoroughly and provide comprehensive information",
            backstory="You are an expert researcher with a talent for finding detailed information on any subject.",
            verbose=True,
            llm=gemini_llm,
            tools=tools
        )
        
        # Create analyst agent with verbose enabled
        progress_placeholder.info("Creating information analyst agent...")
        analyst = Agent(
            role="Information Analyst",
            goal=f"Analyze research findings on {full_topic} and extract key insights",
            backstory="You are a skilled analyst with expertise in synthesizing information and identifying patterns.",
            verbose=True,
            llm=gemini_llm,
            tools=tools
        )
        
        # Create writer agent with verbose enabled
        progress_placeholder.info("Creating content writer agent...")
        writer = Agent(
            role="Content Writer",
            goal=f"Create a well-structured, informative report on {full_topic}",
            backstory="You are a talented writer with a knack for clarity and engaging content.",
            verbose=True,
            llm=gemini_llm
        )
        
        # Adjust task descriptions based on whether we're using search tools
        search_instruction = ""
        if search_provider not in ["No Search Tool (Use LLM Knowledge Only)"]:
            search_instruction = "Use the search tool to gather comprehensive information including recent developments, key concepts, historical context, and relevant statistics. Verify information from multiple sources when possible."
        
        # Create research task
        progress_placeholder.info("Defining research tasks...")
        research_task = Task(
            description=f"Research the topic: {full_topic}. {search_instruction}",
            agent=researcher,
            expected_output="Detailed research findings with all relevant information and sources."
        )
        
        # Create analysis task
        verify_instruction = ""
        if search_provider not in ["No Search Tool (Use LLM Knowledge Only)"]:
            verify_instruction = "Use the search tool to verify or expand on information as needed."
            
        analysis_task = Task(
            description=f"Analyze the research findings on {full_topic}. Identify key trends, patterns, insights, and implications. {verify_instruction}",
            agent=analyst,
            expected_output="In-depth analysis with key insights, trends, and interpretation of the research findings.",
            context=[research_task]
        )
        
        # Create writing task
        writing_task = Task(
            description=f"Using the research and analysis, create a comprehensive report on {full_topic}. The report should be well-structured, informative, and accessible to a general audience.",
            agent=writer,
            expected_output="A complete, well-structured report on the topic with all key information presented clearly.",
            context=[analysis_task]
        )
        
        # Create crew
        progress_placeholder.info("Assembling AI research crew...")
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            verbose=True,
            process=Process.sequential
        )
        
        # Run the crew
        progress_placeholder.warning("Research in progress... This may take several minutes.")
        result = crew.kickoff()
        progress_placeholder.success("Research completed successfully!")
        
        # Restore original stdout
        sys.stdout = stdout_capture.terminal_stdout
        
        # Extract the content as string from CrewOutput object
        if hasattr(result, 'raw'):
            return result.raw
        else:
            # In case the structure changes, try to get a string representation
            return str(result)
        
    except Exception as e:
        # Make sure to restore stdout in case of error
        if 'stdout_capture' in locals():
            sys.stdout = stdout_capture.terminal_stdout
            
        import traceback
        error_details = traceback.format_exc()
        progress_placeholder.error("Research failed!")
        return f"ERROR: {str(e)}\n\n{error_details}"

# Run research when button is clicked
if start_research:
    # Check for API keys in session state, NOT in environment variables
    if not st.session_state.gemini_api_key:
        progress_placeholder.error("Please enter your Gemini API key in the sidebar")
    elif search_provider == "Google Search":
        if not st.session_state.google_search_api_key:
            progress_placeholder.error("Please enter your Google Search API Key in the sidebar")
        elif not st.session_state.google_search_cx:
            progress_placeholder.error("Please enter your Google Custom Search Engine ID (CX) in the sidebar")
        else:
            with results_container:
                with st.spinner("Running research..."):
                    research_result = run_crewai_research(
                        topic=research_topic,
                        focus=research_focus if research_focus else None,
                        gemini_api_key=st.session_state.gemini_api_key,
                        search_provider=search_provider,
                        google_search_api_key=st.session_state.google_search_api_key if search_provider == "Google Search" else None,
                        google_search_cx=st.session_state.google_search_cx if search_provider == "Google Search" else None,
                        model=gemini_model,
                        temp=temperature,
                        show_details=show_agent_details
                    )
                    st.markdown("## Research Results")
                    st.markdown(research_result if research_result else ':red[No results or an error occurred. Please check your API keys and try again.]')
    else:
        with results_container:
            with st.spinner("Running research..."):
                research_result = run_crewai_research(
                    topic=research_topic,
                    focus=research_focus if research_focus else None,
                    gemini_api_key=st.session_state.gemini_api_key,
                    search_provider=search_provider,
                    google_search_api_key=st.session_state.google_search_api_key if search_provider == "Google Search" else None,
                    google_search_cx=st.session_state.google_search_cx if search_provider == "Google Search" else None,
                    model=gemini_model,
                    temp=temperature,
                    show_details=show_agent_details
                )
                st.markdown("## Research Results")
                st.markdown(research_result if research_result else ':red[No results or an error occurred. Please check your API keys and try again.]')