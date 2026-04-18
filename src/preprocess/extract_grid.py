"""
extract_grid.py – Segment individual Chinese characters from calligraphy images
and automatically label each crop via PaddleOCR.

Pipeline:
    1. Remove red seals/stamps (HSV colour masking)
    2. Binarise with adaptive thresholding
    3. Connected-component analysis to find character blobs
    4. Merge nearby tiny components (split radicals)
    5. Sort blobs in reading order (right-to-left columns, top-to-bottom)
    6. Crop each character and recognise it with PaddleOCR
    7. Save as  <unicode_char>_<NNN>.png
"""

import cv2
import numpy as np
import os
import glob
import argparse
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Lazy-loaded singleton so we only pay the OCR init cost once.
# ---------------------------------------------------------------------------
_ocr_engine = None


def _get_ocr_engine():
    """Return a shared PaddleOCR engine (initialised on first call)."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        # Try the new PaddleOCR 3.4+ API first, fall back to legacy.
        try:
            _ocr_engine = PaddleOCR(
                lang='chinese_cht',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except TypeError:
            # Older PaddleOCR (<3.4) uses different parameter names.
            _ocr_engine = PaddleOCR(
                lang='chinese_cht',
                use_angle_cls=False,
                show_log=False,
            )
    return _ocr_engine


# ===================================================================
#  Stage 1 – Red stamp / seal removal
# ===================================================================

def remove_red_stamps(img_bgr):
    """
    Return a copy of *img_bgr* with red-ish pixels painted white.

    Works in HSV: red lives at Hue ≈ 0-10° and 160-180°, with high
    Saturation.  We mask those regions and in-paint them with white so
    that downstream binarisation ignores the seals completely.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around 0° in HSV, so we need two ranges.
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    # Dilate the mask slightly to catch anti-aliased edges of stamps.
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)

    clean = img_bgr.copy()
    clean[red_mask > 0] = (255, 255, 255)
    return clean


# ===================================================================
#  Stage 2 – Binarisation
# ===================================================================

def binarise(img_bgr):
    """
    Convert to a clean binary image (white ink on black background).

    Uses adaptive Gaussian thresholding so that uneven lighting and
    paper discolouration do not cause problems.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Block size must be odd; scale it relative to image size so it works
    # for both small phone-photos and giant scans.
    block = max(31, (min(gray.shape) // 20) | 1)  # ensure odd

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=12,
    )
    return binary


# ===================================================================
#  Stage 3 – Connected-component segmentation
# ===================================================================

def _iou_1d(a_start, a_end, b_start, b_end):
    """Intersection-over-union for two 1-D intervals."""
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _merge_boxes(boxes, img_shape):
    """
    Merge small nearby components that are likely split radicals of a
    single character (e.g. 氵 + 寺 → 持).

    Strategy: if a small box overlaps vertically with a close neighbour
    and the gap between them is tiny, merge them.
    """
    if len(boxes) <= 1:
        return boxes

    # Estimate the "typical" character size as the median box dimension.
    areas = [w * h for (x, y, w, h) in boxes]
    median_area = float(np.median(areas))
    median_side = median_area ** 0.5

    # A component is "small" if its area is < 40% of the median.
    small_thresh = median_area * 0.4
    # Maximum horizontal gap to allow merging (fraction of median side).
    gap_thresh = median_side * 0.6

    merged = []
    used = set()

    # Sort by x so we can scan left-to-right for merge candidates.
    indexed = sorted(enumerate(boxes), key=lambda ib: ib[1][0])

    for i, (idx_a, (xa, ya, wa, ha)) in enumerate(indexed):
        if idx_a in used:
            continue
        bx, by, bw, bh = xa, ya, wa, ha  # accumulator

        for j in range(i + 1, len(indexed)):
            idx_b, (xb, yb, wb, hb) = indexed[j]
            if idx_b in used:
                continue

            area_a = bw * bh
            area_b = wb * hb

            # At least one of them should be "small" for a merge to make sense.
            if area_a > small_thresh and area_b > small_thresh:
                continue

            # Check horizontal gap.
            h_gap = max(0, xb - (bx + bw))
            if h_gap > gap_thresh:
                break  # sorted by x, no point looking further right

            # Check vertical overlap.
            v_overlap = _iou_1d(by, by + bh, yb, yb + hb)
            if v_overlap < 0.25:
                continue

            # Merge!
            nx = min(bx, xb)
            ny = min(by, yb)
            nw = max(bx + bw, xb + wb) - nx
            nh = max(by + bh, yb + hb) - ny
            bx, by, bw, bh = nx, ny, nw, nh
            used.add(idx_b)

        used.add(idx_a)
        merged.append((bx, by, bw, bh))

    return merged


def find_character_boxes(binary, img_shape):
    """
    Return a list of (x, y, w, h) bounding boxes – one per character –
    from the binary image, sorted in reading order.
    """
    # Connected-component analysis.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    img_h, img_w = img_shape[:2]
    img_area = img_h * img_w

    boxes = []
    for i in range(1, num_labels):  # skip label 0 (background)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]  # pixel count, not bbox area

        bbox_area = w * h

        # Filter out dust specks (< 0.05% of image) and page borders (> 50%).
        if bbox_area < img_area * 0.0003 or bbox_area > img_area * 0.5:
            continue

        # Demand minimum ink density inside the bounding box.
        roi = binary[y:y + h, x:x + w]
        ink_ratio = np.count_nonzero(roi) / (w * h)
        if ink_ratio < 0.02:
            continue

        # Reject components that are extremely elongated (likely ruled lines).
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 12:
            continue

        boxes.append((x, y, w, h))

    # Merge split radicals.
    boxes = _merge_boxes(boxes, img_shape)

    # ---- Sort in reading order (right-to-left columns, top-to-bottom) ----
    if not boxes:
        return boxes

    # Determine whether the layout is predominantly vertical or horizontal.
    # Heuristic: if the page is taller than wide → vertical columns.
    # Also look at how characters distribute.
    xs = [b[0] + b[2] / 2 for b in boxes]
    ys = [b[1] + b[3] / 2 for b in boxes]
    x_spread = max(xs) - min(xs) if len(xs) > 1 else 0
    y_spread = max(ys) - min(ys) if len(ys) > 1 else 0

    if x_spread < y_spread * 0.3:
        # Essentially a single column – just sort top-to-bottom.
        boxes.sort(key=lambda b: b[1])
        return boxes

    # Group into columns by X-centre proximity.
    boxes_sorted_x = sorted(boxes, key=lambda b: -(b[0] + b[2] / 2))
    columns = []
    cur_col = [boxes_sorted_x[0]]

    # Adaptive column grouping threshold.
    median_w = float(np.median([b[2] for b in boxes]))
    col_thresh = median_w * 0.8

    for box in boxes_sorted_x[1:]:
        bx_center = box[0] + box[2] / 2
        col_center = np.mean([b[0] + b[2] / 2 for b in cur_col])

        if abs(bx_center - col_center) < col_thresh:
            cur_col.append(box)
        else:
            columns.append(cur_col)
            cur_col = [box]
    if cur_col:
        columns.append(cur_col)

    # Inside each column, sort top-to-bottom.  Columns are already right-to-left.
    ordered = []
    for col in columns:
        col.sort(key=lambda b: b[1])
        ordered.extend(col)

    return ordered


# ===================================================================
#  Stage 4 – Crop & recognise
# ===================================================================

def _sanitise_filename(char: str) -> str:
    """
    Turn a single Unicode character into a safe filename fragment.
    Most CJK characters are fine, but we guard against filesystem-
    unfriendly codepoints by falling back to U+XXXX notation.
    """
    # Characters that are problematic in filenames on Windows/macOS/Linux.
    if char in ('\\', '/', ':', '*', '?', '"', '<', '>', '|', '.', ' '):
        return f"U+{ord(char):04X}"
    # Control characters.
    if ord(char) < 32:
        return f"U+{ord(char):04X}"
    return char


def recognise_character(crop_bgr, ocr_engine):
    """
    Use PaddleOCR to recognise a single character from a cropped image.
    Returns (text, confidence).
    """
    try:
        result = ocr_engine.ocr(crop_bgr, det=False, cls=False)
    except TypeError:
        # New PaddleOCR 3.4+ may not accept det/cls kwargs in .ocr()
        result = ocr_engine.ocr(crop_bgr)

    if not result or not result[0]:
        return None, 0.0

    # Handle both old and new PaddleOCR result formats:
    #   Old: [[('text', confidence), ...]]
    #   New: [{'rec_text': 'X', 'rec_score': 0.99, ...}, ...]
    #         or [[bbox, ('text', confidence)], ...]
    top = result[0]
    if isinstance(top, dict):
        # New dict-based format (PaddleOCR 3.4+)
        text = top.get('rec_text', top.get('text', '')).strip()
        conf = float(top.get('rec_score', top.get('score', 0.0)))
    elif isinstance(top, (list, tuple)) and len(top) == 2:
        # Could be [bbox, ('text', conf)] or ('text', conf)
        inner = top[1] if isinstance(top[0], (list, np.ndarray)) else top
        if isinstance(inner, (list, tuple)) and len(inner) == 2:
            text = str(inner[0]).strip()
            conf = float(inner[1])
        else:
            text = str(inner).strip()
            conf = 0.5
    else:
        # Last resort: try to unpack whatever we got
        try:
            text = str(top[0]).strip()
            conf = float(top[1]) if len(top) > 1 else 0.5
        except (TypeError, IndexError, KeyError):
            return None, 0.0

    if len(text) == 0:
        return None, 0.0

    # Return only the first character if multiple were returned.
    return text[0], conf


def crop_character(img_bgr, box, pad_frac=0.08):
    """
    Crop a character from *img_bgr* given its bounding box, adding
    proportional padding and making the crop square.
    """
    x, y, w, h = box
    ih, iw = img_bgr.shape[:2]

    side = max(w, h)
    pad = int(side * pad_frac)
    side_padded = side + 2 * pad

    # Centre of the bounding box.
    cx, cy = x + w // 2, y + h // 2

    # Square region centred on the character.
    x1 = max(0, cx - side_padded // 2)
    y1 = max(0, cy - side_padded // 2)
    x2 = min(iw, x1 + side_padded)
    y2 = min(ih, y1 + side_padded)

    crop = img_bgr[y1:y2, x1:x2]

    # If the crop ended up non-square (near edges), pad with white.
    ch, cw = crop.shape[:2]
    if ch != cw:
        target = max(ch, cw)
        canvas = np.full((target, target, 3), 255, dtype=np.uint8)
        oy, ox = (target - ch) // 2, (target - cw) // 2
        canvas[oy:oy + ch, ox:ox + cw] = crop
        crop = canvas

    return crop


# ===================================================================
#  Main pipeline
# ===================================================================

def _process_single_image(image_file, output_dir, ocr_engine, counters):
    """
    Process one calligraphy page: segment → recognise → save.

    *counters* is a defaultdict(int) tracking how many times each
    character label has been seen so far (across all images), so that
    duplicates get sequential suffixes.

    Returns the number of characters extracted.
    """
    img = cv2.imread(str(image_file))
    if img is None:
        print(f"WARNING: Could not load image at {image_file}. Skipping.")
        return 0

    # 1. Remove red stamps.
    clean = remove_red_stamps(img)

    # 2. Binarise.
    binary = binarise(clean)

    # 3. Find character bounding boxes.
    boxes = find_character_boxes(binary, img.shape)

    if not boxes:
        print(f"  No characters found in {os.path.basename(str(image_file))}")
        return 0

    extracted = 0
    base_name = os.path.splitext(os.path.basename(str(image_file)))[0]

    for box in boxes:
        crop = crop_character(clean, box)

        # 4. Recognise the character.
        char, conf = recognise_character(crop, ocr_engine)

        if char is None or conf < 0.3:
            # Fall back to a generic label if OCR is unsure.
            char_label = f"_unknown_{base_name}_{extracted:03d}"
        else:
            char_label = _sanitise_filename(char)

        # 5. Save with duplicate suffix.
        counters[char_label] += 1
        idx = counters[char_label]
        out_name = f"{char_label}_{idx:03d}.png"
        cv2.imwrite(os.path.join(output_dir, out_name), crop)
        extracted += 1

    print(
        f"  Extracted {extracted} characters from "
        f"{os.path.basename(str(image_file))}"
    )
    return extracted


def extract_characters_from_grid(image_path, output_dir, **_ignored):
    """
    Extract and label individual character images from calligraphy scans.

    Parameters
    ----------
    image_path : str
        Path to a single image **or** a directory of images.
    output_dir : str
        Directory to write the labelled character crops.

    Returns
    -------
    int
        Total number of characters extracted.
    """
    os.makedirs(output_dir, exist_ok=True)
    ocr_engine = _get_ocr_engine()

    # Shared counter dict so duplicate labels across multiple pages
    # get globally unique suffixes.
    counters = defaultdict(int)

    if os.path.isdir(image_path):
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
            image_files.extend(
                glob.glob(os.path.join(image_path, '**', ext), recursive=True)
            )
        image_files.sort()

        if not image_files:
            print(f"No images found in directory {image_path}")
            return 0

        total = 0
        for i, img_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing {img_file}")
            total += _process_single_image(
                img_file, output_dir, ocr_engine, counters
            )
        print(f"\nTotal extracted characters: {total}")
        return total
    else:
        return _process_single_image(
            image_path, output_dir, ocr_engine, counters
        )


# ===================================================================
#  CLI
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract and label individual Chinese characters from "
            "calligraphy scans using OpenCV segmentation + PaddleOCR."
        )
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to a scanned image or directory of images.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the labelled character crops.",
    )
    # Legacy parameters kept for backwards-compat with runner scripts.
    parser.add_argument(
        "--grid_rows", type=int, default=None,
        help="(Ignored) Legacy parameter.",
    )
    parser.add_argument(
        "--grid_cols", type=int, default=None,
        help="(Ignored) Legacy parameter.",
    )
    args = parser.parse_args()

    extract_characters_from_grid(args.image_path, args.output_dir)
