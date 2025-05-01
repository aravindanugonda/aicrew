# AICrew Research Assistant

A powerful multi-agent AI research tool built with CrewAI and Streamlit that leverages Gemini AI models to perform comprehensive research on any topic through coordinated AI agents.

## Features

- **Multi-Agent Research**: Employs a team of specialized AI agents (Researcher, Analyst, and Writer) working collaboratively
- **Multiple Search Providers**:
  - **No Search Tool**: Rely solely on Gemini's built-in knowledge
  - **Serper.dev**: Integrate with Serper.dev for comprehensive web search capabilities
  - **LinkUp.so**: Leverage LinkUp.so for deep search results
- **Interactive UI**: User-friendly Streamlit interface with detailed logging of agent activities
- **Customizable Research**: Optional focus areas for targeted research
- **Advanced LLM Options**: Control temperature and model selection

## Installation

### Prerequisites

- Python 3.9+ 
- pip (Python package manager)

### Step 1: Clone the repository

```bash
git clone https://github.com/aravindanugonda/aicrew.git
cd aicrew
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv crewai-env
```

Activate the virtual environment:

- On Windows:
  ```bash
  crewai-env\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source crewai-env/bin/activate
  ```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

## API Keys Setup

The application requires API keys for the following services:

1. **Google Gemini API Key**: Required for all modes
   - Get your API key from [Google AI Studio](https://ai.google.dev/)

2. **Serper.dev API Key**: Only required if using Serper.dev search
   - Sign up at [Serper.dev](https://serper.dev/) to get an API key

3. **LinkUp.so API Key**: Only required if using LinkUp.so search
   - Sign up at [LinkUp.so](https://linkup.so/) to get an API key

You can provide these keys directly in the app UI or set them as environment variables:

```bash
# Add to your .env file
GEMINI_API_KEY=your_gemini_api_key
SERPER_API_KEY=your_serper_api_key
LINKUP_API_KEY=your_linkup_api_key
```

## Usage

### Running the app

```bash
streamlit run crew_research_app.py --server.headless true --server.address 0.0.0.0
```

This will start the application and open it in your default web browser.

### Using the app

1. Enter your API keys in the sidebar if not already set as environment variables
2. Select a search provider (or no search tool to use only the LLM's knowledge)
3. Enter a research topic
4. Optionally enter specific aspects to focus on
5. Click "Start Research"
6. View the detailed agent interactions and final research results

## Search Provider Options

### No Search Tool (LLM Knowledge Only)

- Uses only the Gemini model's built-in knowledge
- Faster response times
- Best for general knowledge topics or where up-to-date information isn't critical
- No additional API keys required

### Serper.dev

- Provides comprehensive web search capabilities
- Access to recent information
- Good all-purpose search option
- Requires Serper.dev API key

### LinkUp.so

- Specialized search with deep result capabilities
- Option to control depth of search and output formatting
- More detailed search results
- Requires LinkUp.so API key

## Advanced Configuration

In the "Advanced Options" section of the sidebar, you can:

- Adjust temperature (0.0-1.0): Lower values make responses more deterministic, higher values more creative
- Select Gemini model: Choose between Gemini 1.5 Flash (faster) or Gemini 1.5 Pro (more capable)
- Enable/disable detailed agent interactions

## How It Works

1. **Research Specialist**: Gathers comprehensive information on the topic (using search tool if selected)
2. **Information Analyst**: Processes raw information to extract key insights and patterns
3. **Content Writer**: Creates a well-structured final report based on the research and analysis

The agents work sequentially, each building upon the work of the previous agent.

## Requirements

```
crewai
streamlit
python-dotenv
google-generativeai
crewai-tools

```

## License

[MIT License](LICENSE)

## Acknowledgements

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [Streamlit](https://streamlit.io/) for the web interface
- [Google Gemini](https://ai.google.dev/) for the AI language models
- [Serper.dev](https://serper.dev/) and [LinkUp.so](https://linkup.so/) for search APIs

---
