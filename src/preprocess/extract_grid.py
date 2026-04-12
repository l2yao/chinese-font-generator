import cv2
import numpy as np
import os
from pathlib import Path

def extract_characters_from_grid(image_path, output_dir, grid_rows=10, grid_cols=10):
    """
    Extracts individual character images from a scanned grid of handwriting.
    Assumes the user writes inside boxed grids.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to binarize the image and find grid lines
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
                                   
    # Optional morphological operations could go here to solidify lines
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for grid-like rectangles based on area
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    # Simple sort from top-left to bottom-right
    boundingBoxes = sorted(boundingBoxes, key=lambda b: (b[1], b[0]))
    
    extracted_count = 0
    # In a robust implementation, we would align bounding boxes into rows/cols.
    # For now, we perform a naive extraction loop.
    for x, y, w, h in boundingBoxes:
        # Filter noise, expect cells to be roughly square
        if w > 50 and h > 50 and 0.8 < w/h < 1.2:
            cell = img[y:y+h, x:x+w]
            # Save the cropped cell
            cv2.imwrite(os.path.join(output_dir, f"char_{extracted_count}.png"), cell)
            extracted_count += 1
            
    print(f"Extracted {extracted_count} potentials characters to {output_dir}")
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
