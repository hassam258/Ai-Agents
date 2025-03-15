from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.cryptocompare import CryptoCompareTools
from phi.tools.coingecko import CoinGeckoTools
from phi.tools.alpaca import AlpacaCryptoTools
from dotenv import load_dotenv

load_dotenv()

# Create a crypto market data agent
crypto_data_agent = Agent(
    name="Crypto Data Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    role="Gather cryptocurrency market data and analytics",
    tools=[
        CoinGeckoTools(
            price=True,
            historical_data=True,
            market_chart=True,
            coin_info=True
        ),
        CryptoCompareTools(
            price=True,
            historical_data=True,
            top_exchanges=True,
            social_stats=True
        )
    ],
    show_tool_calls=True,
    markdown=True,
    instructions=[
        "Always present data in clean, formatted tables",
        "Include market sentiment analysis when possible",
        "Provide timeframe context for any price data"
    ],
    debug=True
)

# Create a crypto news agent
crypto_news_agent = Agent(
    name="Crypto News Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    role="Research and summarize cryptocurrency news and trends",
    tools=[DuckDuckGo()],
    instructions=[
        "Always include sources for news items",
        "Prioritize reputable crypto news sources",
        "Highlight potential market impact of news events"
    ],
    show_tool_calls=True,
    markdown=True,
    debug=True
)

# Create a crypto trading agent
crypto_trading_agent = Agent(
    name="Crypto Trading Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    role="Execute and monitor cryptocurrency trades",
    tools=[
        AlpacaCryptoTools(
            account_info=True,
            market_data=True,
            trading=True,
            order_management=True
        )
    ],
    instructions=[
        "Always confirm trade parameters before execution",
        "Monitor position performance",
        "Apply risk management protocols for all trades",
        "Present trade execution results clearly"
    ],
    show_tool_calls=True,
    markdown=True,
    debug=True
)

# Create the complete crypto agent team
crypto_agent_team = Agent(
    name="Crypto Analysis & Trading Team",
    team=[crypto_data_agent, crypto_news_agent, crypto_trading_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    show_tool_calls=True,
    markdown=True,
    instructions=[
        "Begin analyses with market overview and recent price action",
        "Include technical and fundamental analysis in recommendations",
        "Always include risk disclosures with trading suggestions",
        "Present data in clear, formatted tables",
        "Cite sources for all news and market information"
    ],
    debug=True
)

# Example usage
crypto_agent_team.print_response("Analyze Bitcoin's recent performance, summarize analyst sentiment, and suggest a potential trading strategy based on current market conditions", stream=True)