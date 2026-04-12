import argparse
import glob
import os
import shutil


def collect_image_paths(input_dir):
    patterns = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    return sorted(image_paths)


def load_labels(labels_path):
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) == 1 and len(lines[0]) > 1:
        # Allow a single line of characters as shorthand label input.
        return list(lines[0])
    return lines


def main():
    parser = argparse.ArgumentParser(
        description='Label sequentially extracted glyph images with character filenames.'
    )
    parser.add_argument('--input_dir', required=True, help='Directory containing extracted glyph images.')
    parser.add_argument('--labels_file', required=True, help='Text file with one character per line, or one line containing all characters in order.')
    parser.add_argument('--output_dir', required=True, help='Directory to write labeled glyph images for zi2zi-JiT.')
    args = parser.parse_args()

    input_dir = args.input_dir
    labels_file = args.labels_file
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    image_paths = collect_image_paths(input_dir)
    labels = load_labels(labels_file)

    if not image_paths:
        raise ValueError(f'No glyph images found in {input_dir}')

    if len(labels) != len(image_paths):
        raise ValueError(
            f'Label count ({len(labels)}) does not match extracted image count ({len(image_paths)}).'
        )

    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        if len(label) != 1:
            raise ValueError(
                f'Label at line {idx+1} is not a single character: {label!r}'
            )

        output_path = os.path.join(output_dir, f'{label}.png')
        if os.path.exists(output_path):
            raise ValueError(
                f'Duplicate character label detected: {label!r}. '
                'Zi2zi-JiT requires one image file per unique character name.'
            )

        shutil.copy2(img_path, output_path)

    print(f'Copied {len(image_paths)} glyphs to {output_dir} with labeled filenames.')


if __name__ == '__main__':
    main()
