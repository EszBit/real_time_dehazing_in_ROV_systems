
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

# pytorch libs
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/test/final_test/")
parser.add_argument("--sample_dir", type=str, default="data/output/final_test_results/finetuned_no_color")
parser.add_argument("--model_name", type=str, default="funiegan")
parser.add_argument("--model_path", type=str, default="models/finetuned_without_color/generator_final2.pth")
parser.add_argument("--model_label", type=str, default="finetuned")  # custom label for CSV log
parser.add_argument("--resolution", type=str, default="1280x768")
parser.add_argument("--device", type=str, default="mps")  # "cpu" or "mps"

args = parser.parse_args()

# === Device Setup ===
device = torch.device(args.device)
print(f"Using device: {device}")

# === Load model architecture ===
if args.model_name.lower() == 'funiegan':
    from nets import funiegan
    model = funiegan.GeneratorFunieGAN()
elif args.model_name.lower() == 'ugan':
    from nets.ugan import UGAN_Nets
    model = UGAN_Nets(base_model='pix2pix').netG
else:
    raise ValueError("Invalid model name")

# === Load weights ===
assert exists(args.model_path), "model not found"
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Loaded model from {args.model_path}")

# === Output folder ===
os.makedirs(args.sample_dir, exist_ok=True)

# === Image transforms ===
img_width, img_height = map(int, args.resolution.split('x'))
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

# === Run test loop ===
times = []
test_files = sorted(glob(join(args.data_dir, "*.*")))

# === Start monitoring ===
cpu_start = psutil.cpu_percent(interval=1)
ram_start = psutil.virtual_memory().percent
print(f"[START] CPU: {cpu_start}%, RAM: {ram_start}%")

for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = inp_img.to(device).unsqueeze(0)
    start = time.time()
    gen_img = model(inp_img)
    times.append(time.time() - start)
    save_image(gen_img.data, join(args.sample_dir, basename(path)), normalize=True)
    print(f"Tested: {path}")

# === Finish timing and monitoring ===
if len(times) > 1:
    total_time = np.sum(times[1:])  # Skip first for stability
    avg_time = np.mean(times[1:])
    avg_fps = 1. / avg_time
    print(f"\nTotal samples: {len(test_files)}")
    print(f"Time taken: {total_time:.2f} sec, Avg FPS: {avg_fps:.2f}")

    cpu_end = psutil.cpu_percent(interval=1)
    ram_end = psutil.virtual_memory().percent
    print(f"[END] CPU: {cpu_end}%, RAM: {ram_end}%")

    # === Save to CSV ===
    csv_path = "benchmark_results.csv"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Device", "Resolution", "FPS", "TimePerImage", "CPU%", "RAM%"])
        writer.writerow([
            args.model_label,
            args.device,
            args.resolution,
            round(avg_fps, 3),
            round(avg_time, 4),
            cpu_end,
            ram_end
        ])
