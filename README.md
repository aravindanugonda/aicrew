# CrewAI Research Assistant

A powerful research assistant application that leverages multiple AI agents to generate comprehensive research reports on any topic.

## Overview

This application uses:
- **CrewAI**: To orchestrate multiple AI agents working together
- **Google Gemini**: For AI language capabilities
- **Streamlit**: For the user interface

## Features

- Multi-agent research process with specialized roles:
  - Research Specialist: Gathers comprehensive information
  - Information Analyst: Identifies key insights and patterns
  - Content Writer: Creates a well-structured final report
- Simple, user-friendly interface
- Customizable research parameters
- Download research results as markdown files
- No data stored persistently - all processing happens in your local session

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/crewai-research-assistant.git
cd crewai-research-assistant
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv crewai-env
source crewai-env/bin/activate  # On Windows: crewai-env\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project directory and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run crew_research_app.py
```

2. Open your web browser and go to the URL shown in the terminal (usually http://localhost:8501)

3. Enter your research topic and any specific focus areas

4. Click "Start Research" and wait for the results (this may take several minutes)

5. Download or copy the generated research report

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/)
2. Create an account if you don't have one
3. Navigate to API keys and create a new key
4. Copy the key and paste it in the app's sidebar or in your `.env` file

## Requirements

- Python 3.7+
- See requirements.txt for all dependencies

## License

[MIT License](LICENSE)

## Contributions

Contributions, issues, and feature requests are welcome!