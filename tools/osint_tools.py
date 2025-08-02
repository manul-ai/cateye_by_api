"""
OSINT Investigation Tools
"""

import os
import json
import base64
import requests
from dotenv import load_dotenv
import google.genai as genai
import pandas as pd
from playwright.async_api import async_playwright

# Import our existing tools
from sam_segmenter import segment_image_to_chunks

load_dotenv()

# Configure Gemini client
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here'))


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


def upload_image_to_imgbb(image_path):
    """Upload local image to imgbb and return the URL"""
    api_key = os.getenv('IMGBB_API_KEY')

    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    url = "https://api.imgbb.com/1/upload"
    payload = {
        'key': api_key,
        'image': image_data
    }

    response = requests.post(url, data=payload)
    result = response.json()

    if result['success']:
        return result['data']['url']
    else:
        raise Exception(f"Failed to upload image: {result}")


def reverse_image_search(image_url):
    """Perform reverse image search using SerpAPI Google Lens"""
    api_key = os.getenv('SERP_API_KEY')

    url = "https://serpapi.com/search"
    params = {
        'engine': 'google_lens',
        'url': image_url,
        'api_key': api_key
    }

    response = requests.get(url, params=params)
    return response.json()


def results_to_dataframe(results):
    """Convert SerpAPI results to pandas DataFrame"""
    if not results or 'visual_matches' not in results:
        return pd.DataFrame()

    visual_matches = results['visual_matches']
    df = pd.DataFrame(visual_matches)

    # Select most relevant columns
    columns_to_keep = ['position', 'title', 'link', 'source', 'thumbnail',
                       'image', 'thumbnail_width', 'thumbnail_height',
                       'image_width', 'image_height']

    # Keep only columns that exist in the DataFrame
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]

    return df


def reverse_search_local_image(image_path):
    """Main function: upload image and perform reverse search"""
    try:
        # Upload image to imgbb
        image_url = upload_image_to_imgbb(image_path)
        print(f"Image uploaded to: {image_url}")

        # Perform reverse image search
        results = reverse_image_search(image_url)

        # Convert to DataFrame
        df = results_to_dataframe(results)

        return {
            'raw_results': results,
            'dataframe': df
        }

    except Exception as e:
        print(f"Error: {e}")
        return None


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
            matches = df.to_dict('records')[:50]  # Top 50 matches
            
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

async def fetch_web_content_playwright(url: str, search_context: str, capture_images: bool = True) -> str:
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
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            # Navigate to the page with timeout
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for dynamic content to load
            await page.wait_for_timeout(3000)
            
            # Extract page metadata
            title = await page.title()
            page_url = page.url
            
            # Extract text content
            text_content = await page.inner_text('body')
            text_content = text_content[:20000]  # Large limit for LLM processing
            
            # Extract all visible text from specific elements
            headings = [await h.inner_text() for h in await page.query_selector_all('h1, h2, h3, h4, h5, h6')]
            links = [{'text': await link.inner_text(), 'href': await link.get_attribute('href')} 
                    for link in await page.query_selector_all('a[href]')][:20]
            
            # Look for location-related content
            location_keywords = ['address', 'location', 'coordinates', 'GPS', 'street', 'city', 'country', 'latitude', 'longitude']
            location_mentions = [keyword for keyword in location_keywords if keyword.lower() in text_content.lower()]
            
            # Extract meta tags
            meta_tags = {}
            for meta in await page.query_selector_all('meta'):
                name = await meta.get_attribute('name') or await meta.get_attribute('property')
                content = await meta.get_attribute('content')
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
                screenshot = await page.screenshot(full_page=True)
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
                
                for i, img in enumerate(await page.query_selector_all('img[src]')):
                    src = await img.get_attribute('src')
                    alt = await img.get_attribute('alt') or ''
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
                            'width': await img.get_attribute('width'),
                            'height': await img.get_attribute('height')
                        })
                        
                        # Try to capture and save individual images
                        if i < 10:  # Limit to first 10 images
                            try:
                                img_element = await page.query_selector(f'img[src="{src}"]')
                                if img_element:
                                    img_screenshot = await img_element.screenshot()
                                    
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
                                        'title': await img.get_attribute('title', ''),
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
                        img_element = await page.query_selector(f'img[src="{img_info["src"]}"]')
                        if img_element:
                            img_screenshot = await img_element.screenshot()
                            small_images_b64.append({
                                'src': img_info['src'],
                                'alt': img_info['alt'],
                                'image_base64': base64.b64encode(img_screenshot).decode()
                            })
                    except:
                        continue  # Skip if image can't be captured
                
                result['captured_images_base64'] = small_images_b64
            
            await browser.close()
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

def analyze_images(image_paths: list[str], prompt: str) -> str:
    """
    Analyzes one or more images from local paths or remote URLs using a multimodal AI model. This universal tool can describe, compare, and query multiple images in a single call. It can process a list of images, analyze them based on the provided prompt, and return a structured JSON response.

    **Capabilities:**
    - **Batch Analysis:** Provide a list of image paths/URLs to be analyzed together.
    - **Comparative Analysis:** Ask the model to compare features across multiple images (e.g., "Which of these images was taken at night?").
    - **Targeted Search:** Ask the model to find specific features within the set of images (e.g., "Find the image that contains a blue car and describe it.").
    - **Universal Querying:** Ask any question about the visual content of the images.

    **Important:** The tool instructs the AI to return a JSON object where the keys are the image paths/URLs you provided and the values are the analysis for each image. This allows for structured data retrieval.

    Args:
        image_paths: A list of strings, where each string is a local file path or a remote image URL.
        prompt: The specific question or instruction for the AI to follow when analyzing the images.

    Returns:
        A JSON string mapping each image path/URL to the corresponding analysis text. If an error occurs, a JSON object with an 'error' key is returned.
    """
    if not image_paths:
        return json.dumps({"error": "No image paths provided."})

    system_prompt = """
You are an expert image analyst. You will be given one or more images and a prompt.
Your task is to follow the user's prompt for all images provided.
For each image, you must identify it by its source path or URL, which will be provided after the image data.
You MUST return your response as a single, valid JSON object.
The keys of the JSON object MUST be the exact image paths or URLs provided.
The values should be your text analysis for the corresponding image, based on the user's prompt.
    """

    contents = [prompt, system_prompt]

    for path in image_paths:
        try:
            if path.startswith(('http://', 'https://')):
                response = requests.get(path, timeout=15)
                response.raise_for_status()
                mime_type = response.headers.get('Content-Type', 'image/jpeg')
                image_data = response.content
            else:
                if not os.path.exists(path):
                    return json.dumps({"error": f"Local image file not found: {path}"})
                with open(path, 'rb') as f:
                    image_data = f.read()
                ext = os.path.splitext(path)[1].lower()
                mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}
                mime_type = mime_map.get(ext, 'application/octet-stream')

            contents.append(
                {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_data).decode()}}
            )
            contents.append(f"Image source: {path}")

        except Exception as e:
            return json.dumps({"error": f"Failed to process image at {path}: {str(e)}"})

    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=contents
        )

        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        try:
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError:
            return json.dumps({
                "error": "The model did not return a valid JSON object.",
                "raw_response": response_text
            })

    except Exception as e:
        return json.dumps({"error": f"Error during Gemini API call: {str(e)}"})