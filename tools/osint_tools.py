"""
OSINT Investigation Tools
"""

import os
import json
import base64
import requests
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.genai as genai
from playwright.sync_api import sync_playwright

# Import our existing tools
from main import reverse_search_local_image
from sam_segmenter import segment_image_to_chunks

load_dotenv()

# Configure Gemini client
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here'))

def compare_images(image_path1: str, image_path2: str, comparison_type: str = "detailed") -> str:
    """
    Compare two images for OSINT investigation to identify similarities, differences, and connections.
    
    Args:
        image_path1: Path to the first image file
        image_path2: Path to the second image file  
        comparison_type: Type of comparison - "detailed", "location", or "quick"
    
    Returns:
        Detailed comparison results as text
    """
    if not os.path.exists(image_path1):
        return f"Error: First image file {image_path1} not found"
    
    if not os.path.exists(image_path2):
        return f"Error: Second image file {image_path2} not found"
    
    if comparison_type == "detailed":
        prompt = """
        Compare these two images for OSINT investigation. Provide a comprehensive analysis:

        VISUAL COMPARISON:
        1. SIMILARITIES: What elements are the same or very similar between the images?
        2. DIFFERENCES: What are the key differences between the images?
        3. TEMPORAL ANALYSIS: Do these appear to be taken at different times? Evidence?
        4. LOCATION ANALYSIS: Are these the same location? Different angles/viewpoints?
        5. PEOPLE: Same people in both images? Different people? Changes in appearance?
        6. OBJECTS: Same objects/vehicles? New or missing items?
        7. ENVIRONMENTAL CHANGES: Weather, lighting, seasonal differences?
        8. ARCHITECTURAL DETAILS: Building changes, construction, modifications?

        INVESTIGATIVE CONCLUSIONS:
        - Are these images from the same location?
        - Time relationship between images (same time, different times, how far apart?)
        - Evidence of staging or manipulation?
        - What can we conclude about the subjects/locations?
        - Confidence level for each conclusion (High/Medium/Low)

        OSINT LEADS:
        - Which image provides better investigative leads?
        - What questions do the differences raise?
        - What additional verification is needed?
        """
    elif comparison_type == "location":
        prompt = """
        Focus specifically on location analysis between these two images:

        LOCATION COMPARISON:
        1. SAME LOCATION INDICATORS: Architectural features, landmarks, unique elements
        2. DIFFERENT LOCATION INDICATORS: Background differences, architectural styles
        3. VIEWPOINT ANALYSIS: Same location but different angles/distances?
        4. GEOLOCATION CLUES: Street signs, building numbers, distinctive features
        5. ENVIRONMENTAL CONTEXT: Vegetation, terrain, urban vs rural

        CONCLUSION:
        - Are these the same location? (Confidence: High/Medium/Low)
        - If same location: What's the spatial relationship?
        - If different locations: How far apart might they be?
        - Best geolocation leads from comparison
        """
    else:  # quick comparison
        prompt = """
        Quick comparison of these two images for OSINT:
        
        1. SAME SCENE: Yes/No - are these the same location/scene?
        2. TIME DIFFERENCE: Do these appear taken at different times?
        3. KEY DIFFERENCES: 3-5 most important differences
        4. KEY SIMILARITIES: 3-5 most important similarities  
        5. INVESTIGATIVE VALUE: Which image is more useful for investigation?
        6. CONCLUSION: Brief summary of relationship between images
        """
    
    try:
        with open(image_path1, 'rb') as f:
            image_data1 = f.read()
        
        with open(image_path2, 'rb') as f:
            image_data2 = f.read()
        
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[
                prompt,
                "First image:",
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_data1).decode()}},
                "Second image:",
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_data2).decode()}}
            ]
        )
        
        return response.text
    except Exception as e:
        return f"Error comparing images: {str(e)}"

def analyze_image(image_path: str, analysis_type: str) -> str:
    """
    Analyze an image using AI vision to extract detailed information for OSINT purposes.
    
    Args:
        image_path: Path to the image file to analyze
        analysis_type: Type of analysis - "general" or "chunk"
    
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
        11. List of hypotheses about exact location (address, landmark, coordinates if possible)
        12. Time/date estimation (season, lighting, shadows, attire)
        13. People identification (describe features, not names)
        14. Vehicle analysis (make, model, license plates, distinctive features)
        15. Technology/equipment visible (cameras, phones, devices)
        16. Security features (CCTV, barriers, access controls)
        Be specific and detailed for investigative purposes.

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
            model='gemini-2.5-pro',
            contents=[
                prompt,
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_data).decode()}}
            ]
        )
        
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def google_search(query: str, num_results: int = 10) -> str:
    """
    Perform Google search using SerpAPI to find web results for OSINT investigation.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 10, max: 100)
    
    Returns:
        JSON string with search results including titles, links, and snippets
    """
    try:
        api_key = os.getenv('SERP_API_KEY')
        if not api_key:
            return json.dumps({"error": "SERP_API_KEY not found in environment variables"})
        
        url = "https://serpapi.com/search"
        params = {
            'engine': 'google',
            'q': query,
            'api_key': api_key,
            'num': min(num_results, 100)  # Limit to max 100 results
        }
        
        response = requests.get(url, params=params)
        results = response.json()
        
        if "error" in results:
            return f"SerpAPI Error: {results['error']}"
        
        search_summary = {
            'query': query,
            'total_results': results.get('search_information', {}).get('total_results', 0),
            'organic_results': []
        }
        
        # Extract organic search results
        if 'organic_results' in results:
            for result in results['organic_results'][:num_results]:
                search_summary['organic_results'].append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'displayed_link': result.get('displayed_link', ''),
                    'position': result.get('position', 0)
                })
        
        # Add related searches if available
        if 'related_searches' in results:
            search_summary['related_searches'] = [
                search.get('query', '') for search in results['related_searches'][:5]
            ]
        
        # Add knowledge graph if available
        if 'knowledge_graph' in results:
            kg = results['knowledge_graph']
            search_summary['knowledge_graph'] = {
                'title': kg.get('title', ''),
                'type': kg.get('type', ''),
                'description': kg.get('description', ''),
                'source': kg.get('source', {}).get('name', '')
            }
        
        return json.dumps(search_summary, indent=2)
        
    except Exception as e:
        return f"Error performing Google search: {str(e)}"

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
    Downloads and saves images locally for further analysis.
    
    Args:
        url: URL to fetch and analyze
        search_context: Context about what we're looking for
    
    Returns:
        Summary of web content relevant to OSINT investigation including saved image files
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse
        import hashlib
        from datetime import datetime
        
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
        
        # Download and save images
        saved_images = []
        images = soup.find_all('img', src=True)
        
        # Create images directory if it doesn't exist
        os.makedirs('downloaded_images', exist_ok=True)
        
        for i, img in enumerate(images[:10]):  # Limit to first 10 images
            try:
                img_url = img.get('src')
                if not img_url:
                    continue
                
                # Convert relative URLs to absolute
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    img_url = urljoin(url, img_url)
                elif not img_url.startswith(('http://', 'https://')):
                    img_url = urljoin(url, img_url)
                
                # Download image
                img_response = requests.get(img_url, headers=headers, timeout=5)
                if img_response.status_code == 200:
                    # Generate unique filename
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
                    
                    # Get file extension from URL or content type
                    parsed_url = urlparse(img_url)
                    file_ext = os.path.splitext(parsed_url.path)[1] or '.jpg'
                    if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                        content_type = img_response.headers.get('content-type', '')
                        if 'png' in content_type:
                            file_ext = '.png'
                        elif 'gif' in content_type:
                            file_ext = '.gif'
                        elif 'webp' in content_type:
                            file_ext = '.webp'
                        else:
                            file_ext = '.jpg'
                    
                    filename = f"img_{timestamp}_{url_hash}_{i}{file_ext}"
                    file_path = os.path.join('downloaded_images', filename)
                    
                    # Save image
                    with open(file_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    saved_images.append({
                        'filename': filename,
                        'local_path': file_path,
                        'source_url': img_url,
                        'alt_text': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'description': f"Image from {url}",
                        'size_bytes': len(img_response.content)
                    })
                    
            except Exception as img_error:
                continue  # Skip failed image downloads
        
        analysis = {
            'url': url,
            'title': title,
            'content_preview': text_content[:500],
            'location_keywords_found': location_mentions,
            'content_length': len(text_content),
            'search_context': search_context,
            'saved_images': saved_images,
            'images_saved_count': len(saved_images)
        }
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        return f"Error fetching web content from {url}: {str(e)}"

def fetch_web_content_playwright(url: str, search_context: str, capture_images: bool = True) -> str:
    """
    Advanced web content fetching using Playwright for complex sites that require JavaScript execution.
    Can capture page screenshots and extract images for LLM analysis.
    
    Args:
        url: URL to fetch and analyze
        search_context: Context about what we're looking for
        capture_images: Whether to capture screenshots and extract images
    
    Returns:
        Comprehensive page dump including text content, images (as base64), and metadata
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            
            # Navigate to the page with timeout
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for dynamic content to load
            page.wait_for_timeout(3000)
            
            # Extract page metadata
            title = page.title()
            page_url = page.url
            
            # Extract text content
            text_content = page.inner_text('body')[:20000]  # Large limit for LLM processing
            
            # Extract all visible text from specific elements
            headings = [h.inner_text() for h in page.query_selector_all('h1, h2, h3, h4, h5, h6')]
            links = [{'text': link.inner_text(), 'href': link.get_attribute('href')} 
                    for link in page.query_selector_all('a[href]')][:20]
            
            # Look for location-related content
            location_keywords = ['address', 'location', 'coordinates', 'GPS', 'street', 'city', 'country', 'latitude', 'longitude']
            location_mentions = [keyword for keyword in location_keywords if keyword.lower() in text_content.lower()]
            
            # Extract meta tags
            meta_tags = {}
            for meta in page.query_selector_all('meta'):
                name = meta.get_attribute('name') or meta.get_attribute('property')
                content = meta.get_attribute('content')
                if name and content:
                    meta_tags[name] = content
            
            result = {
                'url': page_url,
                'original_url': url,
                'title': title,
                'text_content': text_content,
                'headings': headings,
                'links': links,
                'meta_tags': meta_tags,
                'location_keywords_found': location_mentions,
                'search_context': search_context,
                'content_length': len(text_content)
            }
            
            if capture_images:
                # Take a full page screenshot and save it
                screenshot = page.screenshot(full_page=True)
                result['screenshot_base64'] = base64.b64encode(screenshot).decode()
                
                # Save full page screenshot locally
                from datetime import datetime
                import hashlib
                
                # Create images directory if it doesn't exist
                os.makedirs('downloaded_images', exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                screenshot_filename = f"screenshot_{timestamp}_{url_hash}.png"
                screenshot_path = os.path.join('downloaded_images', screenshot_filename)
                
                with open(screenshot_path, 'wb') as f:
                    f.write(screenshot)
                
                # Extract image sources and save them locally
                images = []
                saved_images = []
                
                for i, img in enumerate(page.query_selector_all('img[src]')):
                    src = img.get_attribute('src')
                    alt = img.get_attribute('alt') or ''
                    if src:
                        # Convert relative URLs to absolute
                        if src.startswith('//'):
                            src = 'https:' + src
                        elif src.startswith('/'):
                            from urllib.parse import urljoin
                            src = urljoin(url, src)
                        
                        images.append({
                            'src': src,
                            'alt': alt,
                            'width': img.get_attribute('width'),
                            'height': img.get_attribute('height')
                        })
                        
                        # Try to capture and save individual images
                        if i < 10:  # Limit to first 10 images
                            try:
                                img_element = page.query_selector(f'img[src="{src}"]')
                                if img_element:
                                    img_screenshot = img_element.screenshot()
                                    
                                    # Generate unique filename for each image
                                    img_hash = hashlib.md5(src.encode()).hexdigest()[:8]
                                    img_filename = f"img_{timestamp}_{img_hash}_{i}.png"
                                    img_path = os.path.join('downloaded_images', img_filename)
                                    
                                    with open(img_path, 'wb') as f:
                                        f.write(img_screenshot)
                                    
                                    saved_images.append({
                                        'filename': img_filename,
                                        'local_path': img_path,
                                        'source_url': src,
                                        'alt_text': alt,
                                        'title': img.get_attribute('title', ''),
                                        'description': f"Image from {url} captured with Playwright",
                                        'size_bytes': len(img_screenshot)
                                    })
                            except:
                                continue  # Skip if image can't be captured
                
                result['images'] = images
                result['saved_images'] = saved_images
                result['images_saved_count'] = len(saved_images)
                result['screenshot_saved'] = {
                    'filename': screenshot_filename,
                    'local_path': screenshot_path,
                    'description': f"Full page screenshot from {url}",
                    'size_bytes': len(screenshot)
                }
                
                # Try to capture small images as base64 for direct analysis (kept for backward compatibility)
                small_images_b64 = []
                for img_info in images[:5]:  # Only first 5 images
                    try:
                        img_element = page.query_selector(f'img[src="{img_info["src"]}"]')
                        if img_element:
                            img_screenshot = img_element.screenshot()
                            small_images_b64.append({
                                'src': img_info['src'],
                                'alt': img_info['alt'],
                                'image_base64': base64.b64encode(img_screenshot).decode()
                            })
                    except:
                        continue  # Skip if image can't be captured
                
                result['captured_images_base64'] = small_images_b64
            
            browser.close()
            return json.dumps(result, indent=2)
            
    except Exception as e:
        return f"Error fetching web content with Playwright from {url}: {str(e)}"

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