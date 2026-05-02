import argparse
import glob
import os
from paddleocr import PaddleOCR
from PIL import Image, ImageOps


INK_THRESHOLD = 210
OUTPUT_SIZE = 256
PADDING_RATIO = 0.12


def clamp_box(box, image_size):
    width, height = image_size
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, int(round(x0))))
    y0 = max(0, min(height - 1, int(round(y0))))
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def ink_projection(image, axis):
    gray = ImageOps.grayscale(image)
    width, height = gray.size
    pixels = gray.load()
    if axis == "x":
        return [
            sum(1 for y in range(height) if pixels[x, y] < INK_THRESHOLD)
            for x in range(width)
        ]
    return [
        sum(1 for x in range(width) if pixels[x, y] < INK_THRESHOLD)
        for y in range(height)
    ]


def smooth(values, radius):
    if radius <= 0:
        return values

    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def split_box_by_ink(image, box, n_chars):
    x0, y0, x1, y1 = clamp_box(box, image.size)
    width = x1 - x0
    height = y1 - y0
    vertical_text = height >= width
    axis = "y" if vertical_text else "x"
    line_image = image.crop((x0, y0, x1, y1))
    projection = ink_projection(line_image, axis)
    length = len(projection)
    step = length / n_chars

    if length <= n_chars:
        boundaries = [round(i * step) for i in range(n_chars + 1)]
    else:
        smoothed = smooth(projection, max(1, int(step * 0.04)))
        boundaries = [0]
        for i in range(1, n_chars):
            expected = i * step
            search_radius = max(2, int(step * 0.35))
            start = max(boundaries[-1] + 1, int(expected - search_radius))
            end = min(length - 1, int(expected + search_radius))
            if start >= end:
                boundary = int(round(expected))
            else:
                boundary = min(range(start, end + 1), key=lambda idx: smoothed[idx])
            boundaries.append(boundary)
        boundaries.append(length)

    boxes = []
    for start, end in zip(boundaries, boundaries[1:]):
        if vertical_text:
            boxes.append((x0, y0 + start, x1, y0 + end))
        else:
            boxes.append((x0 + start, y0, x0 + end, y1))
    return boxes


def trim_sparse_edge_ink(image):
    width, height = image.size
    if width <= 2 or height <= 2:
        return image

    col_projection = ink_projection(image, "x")
    row_projection = ink_projection(image, "y")
    max_x_trim = max(1, int(width * 0.12))
    max_y_trim = max(1, int(height * 0.12))
    sparse_col_limit = max(1, int(height * 0.035))
    sparse_row_limit = max(1, int(width * 0.035))

    left = 0
    while left < max_x_trim and col_projection[left] <= sparse_col_limit:
        left += 1

    right = width - 1
    while right > width - max_x_trim - 1 and col_projection[right] <= sparse_col_limit:
        right -= 1

    top = 0
    while top < max_y_trim and row_projection[top] <= sparse_row_limit:
        top += 1

    bottom = height - 1
    while bottom > height - max_y_trim - 1 and row_projection[bottom] <= sparse_row_limit:
        bottom -= 1

    if left >= right or top >= bottom:
        return image
    return image.crop((left, top, right + 1, bottom + 1))


def ink_bbox(image):
    gray = ImageOps.grayscale(image)
    width, height = gray.size
    pixels = gray.load()
    left = width
    top = height
    right = 0
    bottom = 0
    for y in range(height):
        for x in range(width):
            if pixels[x, y] < INK_THRESHOLD:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x + 1)
                bottom = max(bottom, y + 1)

    if left == width:
        return None
    return left, top, right, bottom


def prepare_training_image(image):
    image = trim_sparse_edge_ink(image)
    bbox = ink_bbox(image)
    if bbox is not None:
        image = image.crop(bbox)

    width, height = image.size
    side = max(width, height)
    padding = max(4, int(side * PADDING_RATIO))
    canvas_side = side + padding * 2
    canvas = Image.new("RGB", (canvas_side, canvas_side), "white")
    canvas.paste(image, ((canvas_side - width) // 2, (canvas_side - height) // 2))
    return canvas.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)


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
                    
                sub_boxes = [box] if n_chars == 1 else split_box_by_ink(image, box, n_chars)
                
                for i in range(n_chars):
                    single_char = text[i]
                    sub_box = sub_boxes[i]
                    
                    # Overwrite any existing files with the same character name
                    out_path = os.path.join(output_dir, f'{single_char}.png')
                    
                    # Crop and save individual character
                    cropped_char = image.crop(clamp_box(sub_box, image.size))

                    # Resize to 256x256 as required by zi2zi-JiT
                    resized_char = prepare_training_image(cropped_char)
                    resized_char.save(out_path)
                    print(f"Saved {out_path} (resized to {OUTPUT_SIZE}x{OUTPUT_SIZE})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and label individual Chinese characters.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    # Legacy parameters
    parser.add_argument("--grid_rows", type=int, default=None, help="(Ignored)")
    parser.add_argument("--grid_cols", type=int, default=None, help="(Ignored)")
    args = parser.parse_args()
    
    extract_characters(args.image_path, args.output_dir)
