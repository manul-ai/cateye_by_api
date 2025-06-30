import os
import cv2
import numpy as np
import argparse
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def load_sam_model(model_type="vit_b", **sam_params):
    """Load SAM model with configurable parameters"""
    model_path = f"sam_{model_type}.pth"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Please download it first.")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    
    # Default parameters that control chunk generation
    default_params = {
        'pred_iou_thresh': 0.88,      # Higher = fewer, higher quality masks
        'stability_score_thresh': 0.95, # Higher = more stable masks only
        'crop_n_layers': 0,           # 0 = no crop, higher = more crops
        'crop_n_points_downscale_factor': 1,
        'min_mask_region_area': 100   # Minimum area for a mask (removes tiny objects)
    }
    
    # Override with provided parameters
    default_params.update(sam_params)
    
    mask_generator = SamAutomaticMaskGenerator(sam, **default_params)
    return mask_generator

def segment_image_to_chunks(image_path, output_dir=None, model_type="vit_b", 
                           pred_iou_thresh=0.88, stability_score_thresh=0.95, 
                           min_mask_region_area=100):
    """
    Segment image into semantic chunks using SAM
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save chunks (default: image_name_chunks)
        model_type: SAM model type (vit_b, vit_l, vit_h)
        pred_iou_thresh: Higher = fewer, higher quality masks (0.0-1.0)
        stability_score_thresh: Higher = more stable masks only (0.0-1.0)
        min_mask_region_area: Minimum area for a mask (removes tiny objects)
    
    Returns:
        List of chunk file paths
    """
    # Setup output directory
    if output_dir is None:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"{image_name}_chunks"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Loading SAM model ({model_type})...")
    sam_params = {
        'pred_iou_thresh': pred_iou_thresh,
        'stability_score_thresh': stability_score_thresh,
        'min_mask_region_area': min_mask_region_area
    }
    mask_generator = load_sam_model(model_type, **sam_params)
    
    if mask_generator is None:
        print("Failed to load SAM model")
        return []
    
    print("Generating masks...")
    masks = mask_generator.generate(image_rgb)
    
    print(f"Found {len(masks)} objects")
    
    chunk_paths = []
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        
        # Get bounding box to crop
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Crop original image to bounding box (no masking)
            cropped_chunk = image_rgb[y_min:y_max+1, x_min:x_max+1]
            
            # Save chunk
            chunk_filename = f"chunk_{i:03d}.png"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            chunk_pil = Image.fromarray(cropped_chunk)
            chunk_pil.save(chunk_path)
            chunk_paths.append(chunk_path)
    
    print(f"Saved {len(chunk_paths)} chunks to {output_dir}")
    return chunk_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment image into semantic chunks using SAM')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--output_dir', help='Output directory for chunks')
    parser.add_argument('--model_type', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type (default: vit_b)')
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88,
                       help='Prediction IoU threshold (higher = fewer chunks, default: 0.88)')
    parser.add_argument('--stability_score_thresh', type=float, default=0.95,
                       help='Stability score threshold (higher = more stable chunks, default: 0.95)')
    parser.add_argument('--min_mask_region_area', type=int, default=100,
                       help='Minimum mask area (removes tiny objects, default: 100)')
    
    args = parser.parse_args()
    
    if os.path.exists(args.image_path):
        chunk_paths = segment_image_to_chunks(
            args.image_path, 
            args.output_dir, 
            args.model_type,
            args.pred_iou_thresh,
            args.stability_score_thresh,
            args.min_mask_region_area
        )
        print(f"Generated {len(chunk_paths)} semantic chunks")
    else:
        print("Image file not found!")