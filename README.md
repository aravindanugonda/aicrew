# AICrew Research Agent

A powerful multi-agent AI research tool built with CrewAI and Streamlit that leverages Gemini AI models to perform comprehensive research on any topic through coordinated AI agents.

## Features

- **Multi-Agent Research**: Employs a team of specialized AI agents (Researcher, Analyst, and Writer) working collaboratively
- **Multiple Search Providers**:
  - **No Search Tool**: Rely solely on Gemini's or OpenAI's built-in knowledge
  - **Google Search**: Integrate with Google Search for comprehensive web search capabilities
- **Interactive UI**: User-friendly Streamlit interface
- **Advanced LLM Options**: Control temperature and model selection; model name updates automatically when provider changes

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

1. **Google Gemini API Key**: Required for Gemini provider
   - Get your API key from [Google AI Studio](https://ai.google.dev/)

2. **OpenAI API Key**: Required for OpenAI provider
   - Get your API key from [OpenAI](https://platform.openai.com/)

3. **Google Search API Key and Custom Search Engine ID (CX)**: Only required if using Google Search
   - Get your API key from your Google Cloud Console (enable the Custom Search API)
   - Set up a Custom Search Engine and obtain the CX ID

You can provide these keys directly in the app UI or set them as environment variables:

```bash
# Add to your .env file
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_SEARCH_API_KEY=your_google_search_api_key
GOOGLE_SEARCH_CX=your_google_search_cx
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
3. Select a model provider (Gemini or OpenAI)
4. The model name field updates automatically with a recommended default for the selected provider (you can override it)
5. Enter a research topic
6. Click "Start Research"
7. View the research results

## Search Provider Options

### No Search Tool (LLM Knowledge Only)

- Uses only the selected model provider's built-in knowledge (Gemini or OpenAI)
- Faster response times
- Best for general knowledge topics or where up-to-date information isn't critical
- No additional API keys required (except Gemini or OpenAI key)

### Google Search Integration

You can now use Google Search as a research tool in addition to Gemini's LLM knowledge. To enable Google Search:

1. Select **Google Search** as the Search Provider in the sidebar.
2. Enter your **Google Search API Key** and **Google Custom Search Engine ID (CX)** in the sidebar fields.
   - You can obtain these from your Google Cloud Console and Custom Search Engine setup.
3. If you do not enter these, only the LLM's built-in knowledge will be used.
4. When enabled, the app will use Google Search results to enhance research quality and freshness.

**Note:**
- No data is stored persistently; all processing happens in your local session.
- If you see errors, double-check your API key and CX values.

## Advanced Configuration

In the "Advanced Options" section of the sidebar, you can:

- Adjust temperature (0.0-1.0): Lower values make responses more deterministic, higher values more creative
- Select model provider (Gemini or OpenAI)
- The model name field updates automatically to a recommended default when you change the provider

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
requests
pysqlite3-binary

```

## License

[MIT License](LICENSE)

## Acknowledgements

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [Streamlit](https://streamlit.io/) for the web interface
- [Google Gemini](https://ai.google.dev/) for the AI language models
- [Google Custom Search](https://programmablesearchengine.google.com/) for search API

---
