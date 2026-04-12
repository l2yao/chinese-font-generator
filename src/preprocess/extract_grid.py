import cv2
import numpy as np
import os
import glob
import argparse

def extract_characters_from_grid(image_path, output_dir, grid_rows=None, grid_cols=None):
    """
    Extracts individual character images from a handwriting scan.
    If image_path is a directory, processes all images within it.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isdir(image_path):
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
            image_files.extend(glob.glob(os.path.join(image_path, '**', ext), recursive=True))
        
        if not image_files:
            print(f"No images found in directory {image_path}")
            return 0
            
        total_extracted = 0
        for img_file in image_files:
            total_extracted += _process_single_image(img_file, output_dir)
        print(f"Total extracted characters: {total_extracted}")
        return total_extracted
    else:
        return _process_single_image(image_path, output_dir)

def _process_single_image(image_file, output_dir):
    img = cv2.imread(str(image_file))
    if img is None:
        print(f"WARNING: Could not load image at {image_file}. Skipping.")
        return 0
        
    # 0. Preprocessing: Isolate black ink and destroy red stamps.
    # By using the Red channel (index 2 in BGR) instead of standard grayscale, 
    # red stamps remain extremely bright (similar to the white paper background). 
    # When inverted and thresholded via Otsu, the red stamp is eliminated as background, 
    # perfectly isolating only the dark black calligraphy ink!
    gray = img[:, :, 2]
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 1. Heavily dilate the ink to fuse disconnected strokes of a single character into one blob.
    # We use an ASYMMETRICAL rectangular kernel here!
    # A large horizontal kernel (~150px) easily bridges disconnected left/right character radicals.
    # A smaller vertical kernel (~50px) strictly prevents separate characters stacked vertically from fusing!
    h_kernel_size = max(5, img.shape[1] // 80)
    v_kernel_size = max(5, img.shape[0] // 200)
    kernel = np.ones((v_kernel_size, h_kernel_size), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # 2. Find all discrete blobs
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    
    # 3. Filter for blobs that actually represent handwriting characters
    # (Ignoring the massive page-spanning vertical rulers and tiny dust particles)
    img_area = img.shape[0] * img.shape[1]
    valid_boxes = []
    for x, y, w, h in boundingBoxes:
        area = w * h
        # A valid character shouldn't be microscopic, nor should it be the entire page border
        if img_area * 0.0005 < area < img_area * 0.5:
            # Demand that the blob has real ink inside of it (filtering out ghost boxes)
            blob_thresh = thresh[y:y+h, x:x+w]
            ink_ratio = np.sum(blob_thresh) / (255.0 * w * h)
            if ink_ratio > 0.02:  # At least 2% ink density
                valid_boxes.append((x, y, w, h))
                
    if not valid_boxes:
        print(f"No valid character blobs found in {image_file}")
        return 0
        
    # 4. Group into vertical columns and sort Right-to-Left, Top-to-Bottom
    # (Handling Traditional Chinese manuscript layout)
    # Sort all boxes primarily by X center: Right-to-Left (descending X)
    valid_boxes.sort(key=lambda b: -(b[0] + b[2]//2))
    
    columns = []
    current_col = [valid_boxes[0]]
    
    # A generous grouping threshold (e.g. 5% of page width) 
    # If two characters' X-centers are within this threshold, they belong to the same vertical column
    col_width_thresh = img.shape[1] * 0.05 
    
    for box in valid_boxes[1:]:
        bx_center = box[0] + box[2]//2
        curr_col_center = sum(b[0] + b[2]//2 for b in current_col) / len(current_col)
        
        if abs(bx_center - curr_col_center) < col_width_thresh:
            current_col.append(box) # Add to current vertical column
        else:
            columns.append(current_col) # Close off the current column
            current_col = [box] # Start a new column further left
            
    if current_col:
        columns.append(current_col)
        
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    extracted_count = 0
    
    # Extract them in proper reading sequence!
    for col in columns:
        # Inside the column, sort characters from top to bottom
        col.sort(key=lambda b: b[1])
        
        for (x, y, w, h) in col:
            # Pad the bounding box slightly so we don't clip the edges of the strokes
            pad = max(5, img.shape[1] // 200)
            y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
            
            cell = img[y1:y2, x1:x2]
            out_name = f"{base_name}_char_{extracted_count:03d}.png"
            cv2.imwrite(os.path.join(output_dir, out_name), cell)
            extracted_count += 1
            
    print(f"Extracted {extracted_count} characters from {image_file} (Sorted Right-to-Left, Top-to-Bottom)")
    return extracted_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract individual characters from Asian handwriting manuscripts.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the scanned image or directory of images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted character images.")
    # Leaving rows and cols for backwards compatibility with runner scripts, but ignoring them internally
    parser.add_argument("--grid_rows", type=int, default=10, help="(Ignored) Legacy parameter.")
    parser.add_argument("--grid_cols", type=int, default=10, help="(Ignored) Legacy parameter.")
    args = parser.parse_args()
    
    extract_characters_from_grid(args.image_path, args.output_dir)
