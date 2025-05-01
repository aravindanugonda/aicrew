
# filename: crew_research_app.py
import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv
import traceback

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="CrewAI Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("🔍 CrewAI Research Assistant")
st.markdown("Research any topic using multiple AI agents powered by Gemini")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    
    # Try to get API key from environment variables first
    api_key_env = os.environ.get("GEMINI_API_KEY", "")
    
    # API key input - prefilled if available in environment
    api_key = st.text_input(
        "Google API Key", 
        value=api_key_env,
        type="password",
        help="Enter your Gemini API key from Google AI Studio (or store it in a .env file as GEMINI_API_KEY)"
    )
    
    # Save API key to environment variable
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        if not api_key_env:  # Only show success if it wasn't already in env
            st.success("API key set successfully!")
    
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
            options=["gemini/gemini-2.0-flash", "gemini/gemini-2.0-pro"],
            index=0,
            help="Select which Gemini model to use"
        )
    
    # About section
    st.markdown("### About")
    st.markdown("""
    This application uses:
    - **CrewAI**: To orchestrate multiple AI agents
    - **Google Gemini**: For AI language capabilities
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
results_container = st.container()

# Function to run research with CrewAI
def run_crewai_research(topic, focus=None, api_key=None, model="gemini/gemini-1.5-flash", temp=0.7):
    """
    Run a research task using CrewAI and return the results
    """
    if not api_key:
        return "ERROR: Please enter your Gemini API key in the sidebar."
    
    if not topic:
        return "ERROR: Please enter a research topic."
    
    try:
        # Set up progress message
        progress_placeholder.info("Initializing research agents...")
        
        # Initialize Gemini LLM
        gemini_llm = LLM(
            model=model,
            api_key=api_key,
            temperature=temp,
        )
        
        # Include focus in the topic if provided
        full_topic = f"{topic}{': ' + focus if focus else ''}"
        
        # Create researcher agent
        progress_placeholder.info("Creating research specialist agent...")
        researcher = Agent(
            role="Research Specialist",
            goal=f"Research {full_topic} thoroughly and provide comprehensive information",
            backstory="You are an expert researcher with a talent for finding detailed information on any subject.",
            verbose=True,
            llm=gemini_llm
        )
        
        # Create analyst agent
        progress_placeholder.info("Creating information analyst agent...")
        analyst = Agent(
            role="Information Analyst",
            goal=f"Analyze research findings on {full_topic} and extract key insights",
            backstory="You are a skilled analyst with expertise in synthesizing information and identifying patterns.",
            verbose=True,
            llm=gemini_llm
        )
        
        # Create writer agent
        progress_placeholder.info("Creating content writer agent...")
        writer = Agent(
            role="Content Writer",
            goal=f"Create a well-structured, informative report on {full_topic}",
            backstory="You are a talented writer with a knack for clarity and engaging content.",
            verbose=True,
            llm=gemini_llm
        )
        
        # Create research task
        progress_placeholder.info("Defining research tasks...")
        research_task = Task(
            description=f"Research the topic: {full_topic}. Focus on gathering comprehensive information including recent developments, key concepts, historical context, and relevant statistics.",
            agent=researcher,
            expected_output="Detailed research findings with all relevant information and sources."
        )
        
        # Create analysis task
        analysis_task = Task(
            description=f"Analyze the research findings on {full_topic}. Identify key trends, patterns, insights, and implications.",
            agent=analyst,
            expected_output="In-depth analysis with key insights, trends, and interpretation of the research findings."
        )
        
        # Create writing task
        writing_task = Task(
            description=f"Using the research and analysis, create a comprehensive report on {full_topic}. The report should be well-structured, informative, and accessible to a general audience.",
            agent=writer,
            expected_output="A complete, well-structured report on the topic with all key information presented clearly."
        )
        
        # Create crew
        progress_placeholder.info("Assembling AI research crew...")
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            verbose=True,
            process=Process.sequential  # Tasks will be completed in order
        )
        
        # Run the crew
        progress_placeholder.warning("Research in progress... This may take several minutes.")
        result = crew.kickoff()
        progress_placeholder.success("Research completed successfully!")
        
        # Extract the content as string from CrewOutput object
        if hasattr(result, 'raw'):
            return result.raw
        else:
            # In case the structure changes, try to get a string representation
            return str(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        progress_placeholder.error("Research failed!")
        return f"ERROR: {str(e)}\n\n{error_details}"

# Run research when button is clicked
if start_research:
    if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
        progress_placeholder.error("Please enter your Gemini API key in the sidebar")
    else:
        with results_container:
            with st.spinner("Running research..."):
                research_result = run_crewai_research(
                    topic=research_topic,
                    focus=research_focus if research_focus else None,
                    api_key=os.environ["GEMINI_API_KEY"],
                    model=gemini_model,
                    temp=temperature
                )
                
                # Display results
                st.markdown("## Research Results")
                st.markdown(research_result)
                
                # Convert the result to string for download button
                result_str = str(research_result)
                
                # Add download button for the results
                st.download_button(
                    label="Download Results",
                    data=result_str,
                    file_name=f"{research_topic.replace(' ', '_')}_research.md",
                    mime="text/markdown"
                )

# Instructions
if not start_research:
    with results_container:
        st.info("""
        ## How to Use This Tool
        
        1. Enter your Google Gemini API key in the sidebar (or add it to a .env file)
        2. Type your research topic in the input field
        3. Optionally specify aspects to focus on
        4. Click "Start Research" to begin
        
        The system will use three AI agents working together:
        - **Research Specialist**: Gathers comprehensive information
        - **Information Analyst**: Identifies key insights and patterns
        - **Content Writer**: Creates a well-structured final report
        
        Research may take several minutes to complete depending on the topic.
        """)

# Add setup instructions at the bottom
with st.expander("Local Setup Instructions"):
    st.markdown("""
    ### How to Run This App Locally
    
    1. **Install required packages**:
       ```bash
       pip install streamlit crewai google-generativeai python-dotenv
       ```
       
    2. **Set up your .env file** (optional but recommended):
       Create a file named `.env` in the same directory as this script with:
       ```
       GEMINI_API_KEY=your_api_key_here
       ```
       
    3. **Save this code** as `crew_research_app.py`
    
    4. **Run the app**:
       ```bash
       streamlit run crew_research_app.py
       ```
       
    5. **Get a Gemini API key**:
       - Go to [Google AI Studio](https://makersuite.google.com/)
       - Create an account if you don't have one
       - Navigate to API keys and create a new key
       - Copy the key and paste it in the sidebar of this app (or in your .env file)
    """)
