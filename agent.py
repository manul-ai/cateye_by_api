"""
OSINT Assistant Agent with Custom Functions
"""

import os
from google.adk.agents import Agent
from prompt import OSINT_AGENT_PROMPT

# Import our OSINT tools as regular Python functions
from tools.osint_tools import (
    analyze_image,
    compare_images,
    google_lens_search, 
    segment_image_sam,
    fetch_web_content,
    fetch_web_content_playwright,
    write_investigation_report,
    google_search
)

# Define ADK-compatible function wrappers
def analyze_image_adk(image_path: str, analysis_type: str) -> str:
    """
    Analyze an image using AI vision to extract detailed information for OSINT purposes.
    This tool will use the Gemini model for analysis with detailed prompts.
    
    Args:
        image_path: Path to the image file to analyze
        analysis_type: Type of analysis - general or chunk
    
    Returns:
        Detailed analysis results as text
    """
    return analyze_image(image_path, analysis_type)

def compare_images_adk(image_path1: str, image_path2: str, comparison_type: str = "detailed") -> str:
    """
    Compare two images for OSINT investigation to identify similarities, differences, and connections.
    Useful for verifying if images are from the same location, time period, or identifying changes over time.
    
    Args:
        image_path1: Path to the first image file
        image_path2: Path to the second image file
        comparison_type: Type of comparison - "detailed", "location", or "quick"
    
    Returns:
        Comprehensive comparison analysis including similarities, differences, temporal analysis, and investigative conclusions
    """
    return compare_images(image_path1, image_path2, comparison_type)

def google_lens_search_adk(image_path: str) -> str:
    """
    Perform reverse image search using Google Lens to find similar images and identify objects/locations.
    
    Args:
        image_path: Path to the image file to search
    
    Returns:
        JSON string with search results
    """
    return google_lens_search(image_path)

def segment_image_sam_adk(image_path: str, quality_threshold: float, min_area: int) -> str:
    """
    Segment an image into semantic chunks using SAM for detailed analysis.
    
    Args:
        image_path: Path to the image file to segment
        quality_threshold: Quality threshold between 0.8-0.98
        min_area: Minimum area for chunks
    
    Returns:
        JSON string with segmentation results
    """
    return segment_image_sam(image_path, quality_threshold, min_area)

def fetch_web_content_adk(url: str, search_context: str) -> str:
    """
    Fetch and analyze web content from URLs found in search results.
    It will use standard tool like requests and BeautifulSoup for analysis. It may not be suitable for complex web content.
    For JavaScript-based websites, consider using other tools. But this tool should be your first choice.
    
    Args:
        url: URL to fetch and analyze
        search_context: Context about what we're looking for
    
    Returns:
        Summary of web content relevant to OSINT investigation
    """
    return fetch_web_content(url, search_context)

def fetch_web_content_playwright_adk(url: str, search_context: str, capture_images: bool = True) -> str:
    """
    Advanced web content fetching using Playwright for complex sites that require JavaScript execution.
    Use this tool when the standard fetch_web_content_adk fails or for JavaScript-heavy websites.
    Can capture page screenshots and extract images for LLM analysis.
    
    Args:
        url: URL to fetch and analyze
        search_context: Context about what we're looking for
        capture_images: Whether to capture screenshots and extract images (default: True)
    
    Returns:
        Comprehensive page dump including text content (up to 20000 chars), images (as base64), and metadata
    """
    return fetch_web_content_playwright(url, search_context, capture_images)

def write_investigation_report_adk(
    image_path: str, 
    findings: str, 
    confidence_assessment: str,
    investigation_leads: str,
    filename: str
) -> str:
    """
    Write a comprehensive OSINT investigation report to a file.
    
    Args:
        image_path: Path to the investigated image
        findings: Main investigation findings
        confidence_assessment: Confidence levels for findings
        investigation_leads: Additional leads for investigation  
        filename: Output filename for the report
    
    Returns:
        Success message with file path
    """
    return write_investigation_report(image_path, findings, confidence_assessment, investigation_leads, filename)



# Create the root agent with custom OSINT tools
root_agent = Agent(
    name="OSINT_Assistant",
    model='gemini-2.5-pro',
    description="An expert OSINT investigator specializing in image analysis and geolocation using advanced AI vision, reverse image search, and web verification.",
    instruction=OSINT_AGENT_PROMPT,
    tools=[
        analyze_image_adk,
        compare_images_adk,
        google_lens_search_adk,
        segment_image_sam_adk,
        fetch_web_content_adk,
        fetch_web_content_playwright_adk,
        write_investigation_report_adk,
        google_search
    ],
)

