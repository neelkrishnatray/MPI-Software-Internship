#Libraries--------------------------------------------------------
from google import genai
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
import os
import asyncio
from dotenv import load_dotenv
#Accessing API Key------------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
#Retry configuration----------------------------------------------
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)
#Creating Agent---------------------------------------------------
trend_Agent = Agent(
    name = "Trend_assistant",
    model = Gemini(
        model = "gemini-2.0-flash",
        api_key = api_key,
        retry_options = retry_config
    ),
    description = "A simple Agent that searches for trends on given ageing intervention",
    instruction = "You are a helpfull trend assistant which uses google search to look up the current trends in context to the given ageing intervention. Do not make up sources and do not create new Facts. Make sure to deliver the sources of found Information",
    tools = [google_search],
)
#Creating a runner------------------------------------------------
runner = InMemoryRunner(agent=trend_Agent)
#creating a response----------------------------------------------
async def main(): 
    response = await runner.run_debug(
    "What are the current trends to rapamycin longevity"
    )
    print(response.text)
if __name__ == "__main__": 
    asyncio.run(main())
