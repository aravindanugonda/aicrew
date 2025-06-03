from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Type
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
import requests
import json
import re
import io
import sys

class SearchProvider(Enum):
    NO_SEARCH = "No Search Tool (Use LLM Knowledge Only)"
    GOOGLE_SEARCH = "Google Search"

@dataclass
class AppConfig:
    gemini_api_key: str = ""
    openai_api_key: str = ""

    google_search_api_key: str = ""
    google_search_cx: str = ""
    search_provider: SearchProvider = SearchProvider.NO_SEARCH
    temperature: float = 0.7
    model_provider: str = "Gemini"
    model_name: str = "gemini/gemini-2.0-flash-lite"
    # show_agent_details removed
    crew_type: str = "research"

class GoogleSearchInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str = Field(description="The search query to perform")

class GoogleSearchTool(BaseTool):
    name: str = "google_search"
    description: str = "Search the web using Google Custom Search API"
    args_schema: Type[BaseModel] = GoogleSearchInput
    _api_key: str = PrivateAttr()
    _cx: str = PrivateAttr()

    def __init__(self, api_key: str, cx: str, **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._cx = cx

    def _run(self, query: str) -> str:
        if not self._api_key or not self._cx:
            return "Error: Google Search not configured"
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {"key": self._api_key, "cx": self._cx, "q": query}
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return self._format_search_results(response.json())
        except Exception as e:
            return f"Search error: {str(e)}"

    def _format_search_results(self, results: Dict[str, Any]) -> str:
        if "items" not in results:
            return "No results found"
        
        formatted = "Google Search Results:\n\n"
        for i, item in enumerate(results["items"], 1):
            title = item.get("title", "No title")
            link = item.get("link", "No link")
            snippet = item.get("snippet", "No description")
            formatted += f"{i}. {title}\n   URL: {link}\n   Description: {snippet}\n\n"
        return formatted

class CrewOrchestrator:
    """Base class for crew orchestration with common functionality"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.tools: List[BaseTool] = []
        self.setup_tools()

    def setup_tools(self):
        """Setup search tools based on configuration"""
        if self.config.search_provider == SearchProvider.GOOGLE_SEARCH:
            self.tools.append(GoogleSearchTool(
                api_key=self.config.google_search_api_key,
                cx=self.config.google_search_cx
            ))

    def create_llm(self) -> LLM:
        """Create configured LLM instance"""
        if self.config.model_provider == "Gemini":
            return LLM(
                model=self.config.model_name,
                api_key=self.config.gemini_api_key,
                temperature=self.config.temperature,
            )
        elif self.config.model_provider == "OpenAI":
            return LLM(
                model=self.config.model_name,
                api_key=self.config.openai_api_key,
                temperature=self.config.temperature,
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.config.model_provider}")

    def create_crew(self, topic: str) -> Crew:
        """Must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement create_crew")

    def execute(self, topic: str) -> str:
        """Execute crew workflow"""
        crew = self.create_crew(topic)
        result = crew.kickoff()
        return str(result)

class ResearchCrew(CrewOrchestrator):
    """Specialized crew for research tasks"""
    def create_crew(self, topic: str) -> Crew:
        llm = self.create_llm()
        
        # Create research agents
        researcher = Agent(
            role="Principal Research Specialist",
            goal=f"Conduct an exhaustive, source-verified literature and data review on {topic}.",
            backstory="A PhD-level researcher celebrated for rigorous methodology, advanced search strategies, "
                      "and the ability to surface hidden insights from vast information pools.",
            verbose=True,
            llm=llm,
            tools=self.tools
        )
        
        analyst = Agent(
            role="Senior Insight Analyst",
            goal=f"Synthesize the collected material on {topic} to reveal patterns, correlations, and knowledge gaps.",
            backstory="Former intelligence analyst skilled in critical thinking, bias detection, and crafting data-driven narratives.",
            verbose=True,
            llm=llm,
            tools=self.tools
        )
        
        writer = Agent(
            role="Lead Content Strategist & Writer",
            goal=f"Transform the analysis on {topic} into a clear, engaging, publication-ready report for an informed lay audience.",
            backstory="Award-winning science communicator known for turning complex findings into compelling stories.",
            verbose=True,
            llm=llm
        )
        
        # Create research tasks
        research_task = Task(
            description=f"Perform a comprehensive literature and data search on {topic}, citing at least 10 high-quality, authoritative sources.",
            agent=researcher,
            expected_output="Markdown dossier containing:\n"
                            "• Annotated bibliography\n"
                            "• Bullet-point summary of key findings\n"
                            "• Relevant statistics or data tables"
        )
        
        analysis_task = Task(
            description=f"Critically evaluate the research dossier on {topic}, extract insights, spot trends, and highlight unanswered questions.",
            agent=analyst,
            expected_output="Insight report including:\n"
                            "• Key themes and patterns\n"
                            "• Implications and potential applications\n"
                            "• Identified knowledge gaps",
            context=[research_task]
        )
        
        writing_task = Task(
            description=f"Draft a well-structured, reader-friendly report on {topic} that integrates the analyst’s insights with narrative flow.",
            agent=writer,
            expected_output="Final report (≈1,500 words) featuring:\n"
                            "• Executive summary\n"
                            "• Main body with sub-headings\n"
                            "• Conclusion and future outlook\n"
                            "• Proper in-text citations and reference list",
            context=[analysis_task]
        )
        
        return Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            verbose=True,
            process=Process.sequential
        )

class StockAnalysisCrew(CrewOrchestrator):
    """Specialized crew for stock market analysis"""
    def create_crew(self, topic: str) -> Crew:
        llm = self.create_llm()
        
        researcher = Agent(
            role="Equity Research Analyst",
            goal=f"Gather macro-economic, sector-specific, and company-level information relevant to {topic}.",
            backstory="Chartered Financial Analyst (CFA) with 10+ years in sell-side equity research covering multiple sectors.",
            verbose=True,
            llm=llm,
            tools=self.tools
        )
        
        analyst = Agent(
            role="Senior Valuation Specialist",
            goal=f"Perform fundamental and technical valuation of {topic}, benchmarking against peers and historical performance.",
            backstory="Excel-savvy analyst specializing in DCF, comparables, and quantitative factor models.",
            verbose=True,
            llm=llm
        )
        
        strategist = Agent(
            role="Portfolio Strategy Lead",
            goal=f"Convert the valuation of {topic} into an actionable investment thesis with clear risk management guidelines.",
            backstory="Former buy-side strategist managing multi-asset portfolios, adept at blending qualitative and quantitative signals.",
            verbose=True,
            llm=llm
        )
        
        research_task = Task(
            description=f"Compile a market intelligence brief on {topic}: recent news, financial statements, competitive landscape, and macro drivers.",
            agent=researcher,
            expected_output="Market research brief containing:\n"
                            "• Company overview\n"
                            "• Latest financial highlights\n"
                            "• Sector and macro context\n"
                            "• Source links"
        )
        
        analysis_task = Task(
            description=f"Create a detailed valuation model and technical assessment for {topic}.",
            agent=analyst,
            expected_output="Valuation report including:\n"
                            "• DCF summary table\n"
                            "• Key multiples vs. peers\n"
                            "• Technical trend analysis (charts optional description)\n"
                            "• Bull, base, bear price targets",
            context=[research_task]
        )
        
        strategy_task = Task(
            description=f"Draft an investment strategy memo for {topic} outlining entry/exit levels, position sizing, and risk scenarios.",
            agent=strategist,
            expected_output="Strategy memo with:\n"
                            "• Investment thesis & catalysts\n"
                            "• Recommended trade setup\n"
                            "• Stop-loss / take-profit levels\n"
                            "• Risk factors and mitigation",
            context=[analysis_task]
        )
        
        return Crew(
            agents=[researcher, analyst, strategist],
            tasks=[research_task, analysis_task, strategy_task],
            verbose=True,
            process=Process.sequential
        )
