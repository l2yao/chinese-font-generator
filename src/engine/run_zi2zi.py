import os
import subprocess
import glob

# This script acts as a bridge wrapper between our Preprocessing/Vectorization pipeline 
# and the specialized `zi2zi-JiT` Diffusion Transformer repository.

ZI2ZI_REPO = "zi2zi-JiT"

def setup_zi2zi():
    """Clones the original repository if not present"""
    if not os.path.exists(ZI2ZI_REPO):
        print(f"Cloning zi2zi-JiT into {ZI2ZI_REPO}...")
        subprocess.run(["git", "clone", "https://github.com/kaonashi-tyc/zi2zi-JiT.git"])
    else:
        print("zi2zi-JiT repository already exists.")

def run_fine_tuning(dataset_dir, output_dir, pretrained_weights="zi2zi-JiT-B-16.pth"):
    """
    Executes the LoRA fine-tuning script provided by zi2zi-JiT.
    The user must place the pretrained `.pth` inside the models folder.
    """
    
    # We must format our dataset from `src/preprocess` into the format zi2zi expects.
    # In a full integration, we'd copy our cropped images into zi2zi's `data_path` folder.
    
    cmd = [
        "python", f"{ZI2ZI_REPO}/lora_single_gpu_finetune_jit.py",
        "--data_path", dataset_dir,
        "--output_dir", output_dir,
        "--base_checkpoint", f"{ZI2ZI_REPO}/models/{pretrained_weights}",
        "--model", "JiT-B/16",
        "--lora_r", "32",
        "--lora_alpha", "32",
        "--epochs", "200", 
        "--batch_size", "16" # Requires ~4GB VRAM
    ]
    
    print("Initiating LoRA Finetuning:")
    print(" ".join(cmd))
    # subprocess.run(cmd)

def run_generation(finetuned_checkpoint, output_dir):
    """
    Spits out thousands of stylized characters utilizing the fast `ab2` sampler from zi2zi.
    """
    cmd = [
        "python", f"{ZI2ZI_REPO}/generate_chars.py",
        "--checkpoint", finetuned_checkpoint,
        "--output_dir", output_dir,
        "--sampling_method", "ab2"
    ]
    
    print("Initiating Font Inference:")
    print(" ".join(cmd))
    # subprocess.run(cmd)

if __name__ == "__main__":
    setup_zi2zi()
    print("To trigger fine-tuning, implement run_fine_tuning(...) hooks.")
