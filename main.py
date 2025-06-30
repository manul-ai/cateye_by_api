import os
import base64
import requests
import argparse
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reverse image search using Google Lens')
    parser.add_argument('image_path', help='Path to the image file')
    args = parser.parse_args()
    
    if os.path.exists(args.image_path):
        result = reverse_search_local_image(args.image_path)
        if result:
            print("Reverse image search results:")
            print(f"Found {len(result['dataframe'])} visual matches")
            print("\nDataFrame:")
            print(result['dataframe'].to_string())
            print(f"\nDataFrame shape: {result['dataframe'].shape}")
            print(f"Columns: {list(result['dataframe'].columns)}")
    else:
        print("Image file not found!")