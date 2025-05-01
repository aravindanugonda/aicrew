import os
import sys
import json
import io
from typing import Type, Dict, Any, List, Optional, Callable, Union
import streamlit as st
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
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

# LinkUp Search Tool implementation
class LinkUpSearchInput(BaseModel):
    """Input schema for LinkUp Search Tool."""
    query: str = Field(description="The search query to perform")
    depth: str = Field(default="standard", description="Depth of search: 'standard' or 'deep'")
    output_type: str = Field(default="searchResults", description="Output type: 'searchResults', 'sourcedAnswer', or 'structured'")

# Global variable to store the LinkUp API key for the current execution only
LINKUP_API_KEY = ""

class LinkUpSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information using LinkUp and return comprehensive results"
    args_schema: Type[BaseModel] = LinkUpSearchInput

    def _run(self, query: str, depth: str = "standard", output_type: str = "searchResults") -> str:
        """Execute LinkUp search and return results."""
        try:
            # Use the global API key
            api_key = LINKUP_API_KEY
            
            # Set up request to LinkUp API
            base_url = "https://api.linkup.so/v1/search"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare the request based on LinkUp's API documentation
            data = {
                "q": query,
                "outputType": output_type,
                "depth": depth
            }
            
            # Make the API call
            response = requests.post(
                base_url, 
                headers=headers, 
                json=data
            )
            response.raise_for_status()
            
            results = response.json()
            
            # Format the results
            formatted_text = "Search Results:\n\n"
            
            # Extract results based on LinkUp API response format
            if "results" in results:
                for i, item in enumerate(results["results"], 1):
                    name = item.get("name", "No title")
                    url = item.get("url", "No link")
                    snippet = item.get("snippet", "No description")
                    
                    formatted_text += f"{i}. {name}\n"
                    formatted_text += f"   URL: {url}\n"
                    formatted_text += f"   Description: {snippet}\n\n"
            else:
                # Handle alternative response structure
                formatted_text += "Results structure not recognized. Raw data:\n"
                formatted_text += json.dumps(results, indent=2)
                
            return formatted_text
            
        except Exception as e:
            return f"Error searching with LinkUp: {str(e)}"

# Initialize session state for API keys if they don't exist
# On first load, try to get values from environment variables
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
if "serper_api_key" not in st.session_state:
    st.session_state.serper_api_key = os.environ.get("SERPER_API_KEY", "")
if "linkup_api_key" not in st.session_state:
    st.session_state.linkup_api_key = os.environ.get("LINKUP_API_KEY", "")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    
    # Add search provider selection
    search_provider = st.selectbox(
        "Search Provider",
        options=["No Search Tool (Use LLM Knowledge Only)", "Serper.dev", "LinkUp.so"],
        index=0,
        help="Select which search API to use for research, or use no search tool"
    )
    
    # API key inputs using session state
    gemini_api_key = st.text_input(
        "Google API Key", 
        value=st.session_state.gemini_api_key,
        type="password",
        help="Enter your Gemini API key from Google AI Studio"
    )
    # Store in session state (not in environment variables)
    st.session_state.gemini_api_key = gemini_api_key
    
    # Show the appropriate API key input based on selection
    if search_provider == "Serper.dev":
        serper_api_key = st.text_input(
            "Serper API Key", 
            value=st.session_state.serper_api_key,
            type="password",
            help="Enter your Serper.dev API key for web search capabilities"
        )
        # Store in session state (not in environment variables)
        st.session_state.serper_api_key = serper_api_key
    elif search_provider == "LinkUp.so":
        linkup_api_key = st.text_input(
            "LinkUp API Key", 
            value=st.session_state.linkup_api_key,
            type="password",
            help="Enter your LinkUp.so API key for web search capabilities"
        )
        # Store in session state (not in environment variables)
        st.session_state.linkup_api_key = linkup_api_key
        # Set the global variable for current execution only
        LINKUP_API_KEY = linkup_api_key
    
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
            options=["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro"],
            index=0,
            help="Select which Gemini model to use"
        )
        
        show_agent_details = st.checkbox(
            "Show detailed agent interactions",
            value=True,
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
      - Serper.dev for web search 
      - LinkUp.so for web search
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
def run_crewai_research(topic, focus=None, gemini_api_key=None, search_provider="No Search Tool (Use LLM Knowledge Only)",
                        serper_api_key=None, linkup_api_key=None,
                        model="gemini/gemini-1.5-flash", temp=0.7, show_details=True):
    """
    Run a research task using CrewAI and return the results
    """
    if not gemini_api_key:
        return "ERROR: Please enter your Gemini API key in the sidebar."
    
    if search_provider == "Serper.dev" and not serper_api_key:
        return "ERROR: Please enter your Serper API key in the sidebar."
    
    if search_provider == "LinkUp.so" and not linkup_api_key:
        return "ERROR: Please enter your LinkUp API key in the sidebar."
    
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
        if search_provider == "Serper.dev":
            search_tool = SerperDevTool(api_key=serper_api_key)
            tools = [search_tool]
            progress_placeholder.info("Initializing Serper.dev search tool...")
        elif search_provider == "LinkUp.so":
            # Set the global variable for LinkUp API key
            global LINKUP_API_KEY
            LINKUP_API_KEY = linkup_api_key
            
            # Create the tool without passing the API key
            search_tool = LinkUpSearchTool()
            tools = [search_tool]
            progress_placeholder.info("Initializing LinkUp.so search tool...")
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
    elif search_provider == "Serper.dev" and not st.session_state.serper_api_key:
        progress_placeholder.error("Please enter your Serper API key in the sidebar")
    elif search_provider == "LinkUp.so" and not st.session_state.linkup_api_key:
        progress_placeholder.error("Please enter your LinkUp API key in the sidebar")
    else:
        with results_container:
            with st.spinner("Running research..."):
                research_result = run_crewai_research(
                    topic=research_topic,
                    focus=research_focus if research_focus else None,
                    gemini_api_key=st.session_state.gemini_api_key,
                    search_provider=search_provider,
                    serper_api_key=st.session_state.serper_api_key,
                    linkup_api_key=st.session_state.linkup_api_key,
                    model=gemini_model,
                    temp=temperature,
                    show_details=show_agent_details
                )
                
                # Display results
                st.markdown("## Research Results")
                st.markdown(research_result)