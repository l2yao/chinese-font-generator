# Chinese Font Generator Pipeline

This repository provides an automated pipeline for **Few-Shot Chinese Font Generation**. Given a few samples of a person's handwritten Chinese characters, this framework utilizes the State-of-the-Art [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) (Pixel Space Diffusion Transformers) to learn the user's calligraphy style and synthesize a complete TrueType Font (`.ttf`).

## Repository Map

The project pipeline stitches together three stages:

1. **Preprocess (`src/preprocess/extract_grid.py`)** 
   - Uses PaddleOCR to automatically detect, crop, and label each handwritten character from a scanned page into individual images.
2. **Engine (`src/engine/run_zi2zi.py`)** 
   - Acts as a wrapper over the custom **zi2zi-JiT** architecture.
   - Automatically utilizes `lora_single_gpu_finetune_jit.py` to efficiently teach the Diffusion Transformer your style using limited VRAM.
   - Uses `generate_chars.py` with fast `ab2` samplers to bulk-generate your styling over standard Chinese content vectors.
3. **Build Font (`src/build_font/vectorize.py`)**
   - Synthesizes the output `.png` character images into scalable vector paths.
   - Packages them into `.ttf`.

## Setup Instructions

### Running Locally
To run this pipeline locally, you will need a GPU with at least 4GB-8GB VRAM (for LoRA fine-tuning).

1. Clone this directory.
2. Install dependencies. We recommend using conda as dictated by zi2zi-JiT:
   ```bash
   conda env create -f environment.yaml 
   conda activate zi2zi-jit 
   pip install -r requirements.txt
   ```
3. Run the scripts sequentially:
   ```bash
   python src/preprocess/extract_grid.py 
   python src/engine/run_zi2zi.py
   python src/build_font/vectorize.py
   ```

### Running in Google Colab (Highly Recommended)

If you do not have adequate local GPU hardware, you can easily train this model using a free Google Colab T4 GPU!

A fully contained notebook is provided in the root directory: `Font_Generation_Colab.ipynb`

Simply upload the notebook to [Google Colab](https://colab.research.google.com/), execute the cells sequentially to install the requirements, clone the zi2zi-JiT engine, process your handwriting, and generate your font files.
