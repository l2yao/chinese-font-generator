import cv2
import numpy as np
import os
from pathlib import Path

def extract_characters_from_grid(image_path, output_dir, grid_rows=10, grid_cols=10):
    """
    Extracts individual character images from a scanned grid of handwriting.
    If image_path is a directory, processes all images within it.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isdir(image_path):
        import glob
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
            image_files.extend(glob.glob(os.path.join(image_path, '**', ext), recursive=True))
        
        if not image_files:
            print(f"No images found in directory {image_path}")
            return 0
            
        total_extracted = 0
        for img_file in image_files:
            total_extracted += _process_single_image(img_file, output_dir, grid_rows, grid_cols)
        print(f"Total extracted characters: {total_extracted}")
        return total_extracted
    else:
        return _process_single_image(image_path, output_dir, grid_rows, grid_cols)

def _process_single_image(image_file, output_dir, grid_rows=10, grid_cols=10):
    img = cv2.imread(str(image_file))
    if img is None:
        print(f"WARNING: Could not load image at {image_file}. Skipping.")
        return 0
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
                                   
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    boundingBoxes = sorted(boundingBoxes, key=lambda b: (b[1], b[0]))
    
    extracted_count = 0
    # Use base filename to prevent naming collisions
    base_name = os.path.splitext(os.path.basename(image_file))[0]

    for x, y, w, h in boundingBoxes:
        if w > 50 and h > 50 and 0.8 < w/h < 1.2:
            cell = img[y:y+h, x:x+w]
            out_name = f"{base_name}_char_{extracted_count}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), cell)
            extracted_count += 1
            
    print(f"Extracted {extracted_count} characters from {image_file}")
    return extracted_count

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract characters from a handwriting grid.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the scanned grid image.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted character images.")
    parser.add_argument("--grid_rows", type=int, default=10, help="Number of rows in the grid.")
    parser.add_argument("--grid_cols", type=int, default=10, help="Number of columns in the grid.")
    args = parser.parse_args()
    
    extract_characters_from_grid(args.image_path, args.output_dir, args.grid_rows, args.grid_cols)
