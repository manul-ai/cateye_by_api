"""
OSINT Investigation Tools
"""

import os
import json
import base64
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.genai as genai

# Import our existing tools
from main import reverse_search_local_image
from sam_segmenter import segment_image_to_chunks

load_dotenv()

# Configure Gemini client
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here'))

def analyze_image(image_path: str, analysis_type: str) -> str:
    """
    Analyze an image using AI vision to extract detailed information for OSINT purposes.
    
    Args:
        image_path: Path to the image file to analyze
        analysis_type: Type of analysis - "general", "detailed", or "chunk"
    
    Returns:
        Detailed analysis results as text
    """
    if not os.path.exists(image_path):
        return f"Error: Image file {image_path} not found"
    
    if analysis_type == "general":
        prompt = """
        Analyze this image for OSINT investigation. Provide structured information:
        
        1. LOCATION TYPE: Indoor/Outdoor, Private/Public space
        2. ENVIRONMENT: Urban/Rural/Suburban/Industrial/Natural
        3. PEOPLE: Count, visible characteristics, activities
        4. OBJECTS: Vehicles, signs, landmarks, distinctive items
        5. TEXT: Any readable text, signs, license plates, brands
        6. ARCHITECTURE: Building styles, architectural details
        7. TIME INDICATORS: Season, time of day, weather, lighting
        8. GEOGRAPHIC CLUES: Vegetation, landscape, architectural style
        9. CULTURAL INDICATORS: Language, symbols, cultural markers
        10. NOTABLE DETAILS: Unique or identifying features
        
        Be specific and detailed for investigative purposes.
        """
    elif analysis_type == "detailed":
        prompt = """
        Perform deep OSINT analysis of this image:
        
        IDENTIFICATION PRIORITIES:
        - Exact location (address, landmark, coordinates if possible)
        - Time/date estimation (season, lighting, shadows, attire)
        - People identification (describe features, not names)
        - Vehicle analysis (make, model, license plates, distinctive features)
        - Technology/equipment visible (cameras, phones, devices)
        - Security features (CCTV, barriers, access controls)
        
        INVESTIGATIVE LEADS:
        - Reverse searchable elements (landmarks, signs, distinctive objects)
        - Cross-referenceable details (architectural styles, vegetation, infrastructure)
        - Potential verification sources (official buildings, public spaces)
        
        Provide confidence levels for each conclusion.
        """
    else:  # chunk analysis
        prompt = """
        Analyze this image chunk for OSINT purposes:
        - Main object/element identification
        - Readable text or symbols
        - Brand names, logos, identifying marks
        - Architectural or design details
        - Any geographic or cultural indicators
        - Potential for reverse image search
        
        Be concise but specific for investigation leads.
        """
    
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[
                prompt,
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_data).decode()}}
            ]
        )
        
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def google_lens_search(image_path: str) -> str:
    """
    Perform reverse image search using Google Lens to find similar images and identify objects/locations.
    
    Args:
        image_path: Path to the image file to search
    
    Returns:
        JSON string with search results
    """
    if not os.path.exists(image_path):
        return f"Error: Image file {image_path} not found"
    
    try:
        result = reverse_search_local_image(image_path)
        if result and 'dataframe' in result:
            df = result['dataframe']
            matches = df.to_dict('records')[:15]  # Top 15 matches
            
            search_summary = {
                'total_matches': len(df),
                'top_matches': []
            }
            
            for match in matches:
                search_summary['top_matches'].append({
                    'title': match.get('title', ''),
                    'source': match.get('source', ''),
                    'link': match.get('link', ''),
                    'confidence_rank': match.get('position', 0)
                })
            
            return json.dumps(search_summary, indent=2)
        else:
            return "No search results found"
    except Exception as e:
        return f"Error performing Google Lens search: {str(e)}"

def segment_image_sam(image_path: str, quality_threshold: float, min_area: int) -> str:
    """
    Segment an image into semantic chunks using SAM (Segment Anything Model) for detailed analysis.
    
    Args:
        image_path: Path to the image file to segment
        quality_threshold: Quality threshold (0.8-0.98, higher = fewer chunks)
        min_area: Minimum area for chunks (removes tiny objects)
    
    Returns:
        JSON string with segmentation results
    """
    if not os.path.exists(image_path):
        return f"Error: Image file {image_path} not found"
    
    try:
        chunk_paths = segment_image_to_chunks(
            image_path,
            pred_iou_thresh=quality_threshold,
            min_mask_region_area=min_area
        )
        
        return json.dumps({
            'chunk_count': len(chunk_paths),
            'chunk_paths': chunk_paths[:20],  # Limit to 20 for processing
            'segmentation_parameters': {
                'quality_threshold': quality_threshold,
                'min_area': min_area
            }
        }, indent=2)
    except Exception as e:
        return f"Error segmenting image: {str(e)}"

def fetch_web_content(url: str, search_context: str) -> str:
    """
    Fetch and analyze web content from URLs found in search results.
    
    Args:
        url: URL to fetch and analyze
        search_context: Context about what we're looking for
    
    Returns:
        Summary of web content relevant to OSINT investigation
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract key information
        title = soup.find('title').text if soup.find('title') else "No title"
        
        # Extract text content (first 2000 chars)
        text_content = soup.get_text()[:2000]
        
        # Look for location-related keywords
        location_keywords = ['address', 'location', 'coordinates', 'GPS', 'street', 'city', 'country']
        location_mentions = [keyword for keyword in location_keywords if keyword.lower() in text_content.lower()]
        
        analysis = {
            'url': url,
            'title': title,
            'content_preview': text_content[:500],
            'location_keywords_found': location_mentions,
            'content_length': len(text_content),
            'search_context': search_context
        }
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        return f"Error fetching web content from {url}: {str(e)}"

def write_investigation_report(
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
    try:
        from datetime import datetime
        
        report_content = f"""
OSINT INVESTIGATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================

TARGET IMAGE: {image_path}

INVESTIGATION FINDINGS:
{findings}

CONFIDENCE ASSESSMENT:
{confidence_assessment}

ADDITIONAL INVESTIGATION LEADS:
{investigation_leads}

================================
End of Report
"""
        
        with open(filename, 'w') as f:
            f.write(report_content)
        
        return f"✅ Investigation report successfully written to: {filename}"
        
    except Exception as e:
        return f"❌ Error writing report: {str(e)}"