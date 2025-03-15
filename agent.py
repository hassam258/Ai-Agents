from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

load_dotenv()

agent=Agent(
    model=Groq(id="llama-3.3-70b-versatile")
)
agent.print_response("write a short paragraph on the recent trends in the AI industry")

agent.print_response("write a motivation quote for a data scientist who has experienced layy off") 
