"""
OSINT Assistant Agent
"""

from google.adk import Agent
from google.adk.tools import google_search

from tools.osint_tools import (
    analyze_image,
    google_lens_search,
    segment_image_sam,
    fetch_web_content,
    write_investigation_report
)
from prompt import OSINT_AGENT_PROMPT

# Create function tools
tools = [
    analyze_image,
    google_lens_search,
    segment_image_sam,
    fetch_web_content,
    write_investigation_report,
    google_search,
]

# Create the root agent for ADK web interface
root_agent = Agent(
    name="OSINT_Assistant",
    model='gemini-2.0-flash-exp',
    description="An expert OSINT investigator specializing in image analysis and geolocation. Can analyze images, perform reverse searches, segment images with SAM, and create comprehensive investigation reports.",
    instruction=OSINT_AGENT_PROMPT,
    tools=tools,
)