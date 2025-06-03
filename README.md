# AICrew Orchestrator

AICrew Orchestrator is a Streamlit-based application that orchestrates multiple AI agents (crews) for research and stock analysis tasks. It leverages CrewAI, Google Gemini, and OpenAI models, and supports web search via Google Custom Search API.

## Features

- **Multi-agent orchestration**: Research and Stock Analysis crews, each with specialized roles (researcher, analyst, writer/strategist).
- **Diverse LLM outputs**: Each research task is run with three different temperature settings (0.7, 0.3, 1.0) for broader coverage.
- **Formatting preservation**: Agents are instructed to preserve all tables, code blocks, and markdown formatting throughout the workflow.
- **Conversational output**: Final reports use a friendly, accessible tone and avoid excessive formality or bullet points.
- **Rich markdown rendering**: Results are displayed using Streamlit's markdown renderer for a ChatGPT/Gemini-like experience.
- **Custom monospace font for code**: Code blocks and tables use a more readable monospace font.
- **Configurable via sidebar**: Easily select crew type, model provider, search provider, and enter API keys.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (or use a `.env` file):
   - `GEMINI_API_KEY` (for Google Gemini)
   - `OPENAI_API_KEY` (for OpenAI)
   - `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_CX` (for Google Custom Search, optional)

3. **Run the app**:
   ```bash
   streamlit run crew_research_app.py
   ```

## Usage

- Select the crew type (Research or Stock Analysis) in the sidebar.
- Choose your model provider (Gemini or OpenAI) and enter the relevant API key.
- (Optional) Enable Google Search and provide API credentials for web-augmented research.
- Enter your research topic or stock symbol and click "Start Task".
- Results are shown in rich markdown, preserving tables and formatting.

## Key Changes in This Version

- **Temperature Diversity**: Research is performed at three LLM temperature settings and combined for richer context.

## License
MIT
