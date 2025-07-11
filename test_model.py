"""
Batch inference script for evaluating image enhancement models (FUnIE-GAN or UGAN).

This script loads a pre-trained PyTorch model and processes a folder of test images.
Each image is optionally resized, passed through the model, and the enhanced output is saved.
It also benchmarks average inference time and FPS, logs system resource usage, and optionally
writes results to a CSV file.

Recommended for quickly testing model performance and image quality on CPU or MPS (Apple Silicon).
"""

import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
import psutil
import csv
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/test/")
parser.add_argument("--sample_dir", type=str, default="data/output/test_results")
parser.add_argument("--model_name", type=str, default="funiegan")
parser.add_argument("--model_path", type=str, default="models/finetuned_color/generator_final.pth")
parser.add_argument("--model_label", type=str, default="finetuned")
# use 640x480, 960x576, 1280x768 for best results
parser.add_argument("--resolution", type=str, default="960x576", help="Target resolution in WxH format")
parser.add_argument("--device", type=str, default="mps")  # "mps" or "cpu"
parser.add_argument("--csv_path", type=str, default="results/benchmark_results.csv")
parser.add_argument("--overwrite_csv", action="store_true", help="Clear CSV before logging")
args = parser.parse_args()

# Device Setup
device = torch.device(args.device)
print(f"Using device: {device}")

# Image resolution
if args.resolution is not None:
    try:
        width, height = map(int, args.resolution.lower().split('x'))
    except ValueError:
        raise ValueError("Resolution must be in the format WxH, e.g., 1280x768")
else:
    width, height = None, None

# Load model
if args.model_name.lower() == 'funiegan':
    from nets import funiegan
    model = funiegan.GeneratorFunieGAN()
elif args.model_name.lower() == 'ugan':
    from nets.ugan import UGAN_Nets
    model = UGAN_Nets(base_model='pix2pix').netG
else:
    raise ValueError("Invalid model name")

assert exists(args.model_path), "Model not found!"
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Loaded model from: {args.model_path}")

# Setup folders
os.makedirs(args.sample_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

# Image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Test loop
times = []
test_files = sorted(glob(join(args.data_dir, "*.*")))

cpu_start = psutil.cpu_percent(interval=1)
ram_start = psutil.virtual_memory().percent
print(f"[START] CPU: {cpu_start}%, RAM: {ram_start}%")

for path in test_files:
    img = Image.open(path).convert("RGB")
    if width and height:
        img = img.resize((width, height), Image.BICUBIC)
    inp_img = transform(img).unsqueeze(0).to(device)

    start = time.time()
    gen_img = model(inp_img)
    times.append(time.time() - start)

    save_image(gen_img.data, join(args.sample_dir, basename(path)), normalize=True)
    print(f"Tested: {path}")

# Runtime stats
if len(times) > 1:
    total_time = np.sum(times[1:])  # skip 1st image
    avg_time = np.mean(times[1:])
    avg_fps = 1. / avg_time

    # Consistency check
    inv_time = round(1. / avg_time, 4)
    assert abs(inv_time - avg_fps) < 1e-3, "Inconsistent FPS vs Time/image!"

    cpu_end = psutil.cpu_percent(interval=1)
    ram_end = psutil.virtual_memory().percent

    print("\n=== Test Summary ===")
    print(f"Total images: {len(test_files)}")
    print(f"Avg time/image: {avg_time:.4f} sec")
    print(f"Avg FPS: {avg_fps:.2f}")
    print(f"[END] CPU: {cpu_end}%, RAM: {ram_end}%")
    print("====================\n")

    # CSV logging
    write_header = args.overwrite_csv or not os.path.exists(args.csv_path)
    mode = 'w' if args.overwrite_csv else 'a'
    with open(args.csv_path, mode=mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Device", "FPS", "TimePerImage", "CPU%", "RAM%"])
        writer.writerow([
            args.model_label,
            args.device,
            round(avg_fps, 3),
            round(avg_time, 4),
            cpu_end,
            ram_end
        ])
