import argparse
import glob
import os
from paddleocr import PaddleOCR
from PIL import Image

def extract_characters(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize PaddleOCR with detection enabled
    ocr = PaddleOCR(
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False, 
        use_textline_orientation=False
    )
    
    if os.path.isdir(image_path):
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
            image_files.extend(glob.glob(os.path.join(image_path, '**', ext), recursive=True))
        image_files.sort()
    else:
        image_files = [image_path]

    for img_path in image_files:
        print(f"Processing {img_path}")
        
        # 2. Perform OCR
        result = ocr.predict(img_path)
        
        # 3. Process results and crop
        image = Image.open(img_path).convert('RGB')
        for res in result:
            # Debug info commented out
            # res.save_to_img("output")
            # res.save_to_json("output")
            
            for ind in range(len(res["rec_texts"])):
                text = res["rec_texts"][ind]
                box = res["rec_boxes"][ind]
                
                # Check if it's a multi-character segment
                n_chars = len(text)
                if n_chars == 0:
                    continue
                    
                x0, y0, x1, y1 = box
                width = x1 - x0
                height = y1 - y0
                
                for i in range(n_chars):
                    single_char = text[i]
                    
                    if n_chars == 1:
                        sub_box = box
                    else:
                        if width > height:
                            # Horizontal text line, split along width
                            step = width / n_chars
                            sub_box = (x0 + i * step, y0, x0 + (i + 1) * step, y1)
                        else:
                            # Vertical text line, split along height
                            step = height / n_chars
                            sub_box = (x0, y0 + i * step, x1, y0 + (i + 1) * step)
                    
                    # Overwrite any existing files with the same character name
                    out_path = os.path.join(output_dir, f'{single_char}.png')
                    
                    # Crop and save individual character
                    cropped_char = image.crop(sub_box)
                    cropped_char.save(out_path)
                    print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and label individual Chinese characters.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    # Legacy parameters
    parser.add_argument("--grid_rows", type=int, default=None, help="(Ignored)")
    parser.add_argument("--grid_cols", type=int, default=None, help="(Ignored)")
    args = parser.parse_args()
    
    extract_characters(args.image_path, args.output_dir)
