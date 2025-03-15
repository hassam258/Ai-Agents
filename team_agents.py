from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

web_agent=Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include source"],
    show_tool_calls=True,
    markdown=True,
  
)
finance_agent=Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    role="Get Financial data",
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,stock_fundamentals=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Use Tables to display the data"],
    debug=True
)

agent_team=Agent(
    team=[web_agent,finance_agent],
    
    model=Groq(id="llama-3.3-70b-versatile"),
    # tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,stock_fundamentals=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Always inlcude sources", "Use Tables to display the data"],
    debug=True
)
agent_team.print_response("Summarize analyst recomendations and sahre the latest news for NVDA", stream=True)
