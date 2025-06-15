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

# === PyTorch
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms

# === Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/test/final_test/")
parser.add_argument("--sample_dir", type=str, default="data/output/final_test_results/test_test")
parser.add_argument("--model_name", type=str, default="funiegan")
parser.add_argument("--model_path", type=str, default="models/finetuned_color/generator_final.pth")
parser.add_argument("--model_label", type=str, default="finetuned")
parser.add_argument("--resolution", type=str, default="1280x768")
parser.add_argument("--device", type=str, default="mps")  # or "cpu"
parser.add_argument("--csv_path", type=str, default="results/benchmark_results.csv")
parser.add_argument("--overwrite_csv", action="store_true", help="Clear CSV before logging")
args = parser.parse_args()

# === Device Setup
device = torch.device(args.device)
print(f"Using device: {device}")

# === Load model
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

# === Setup folders
os.makedirs(args.sample_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

# === Image transforms
img_width, img_height = map(int, args.resolution.split('x'))
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

# === Test loop
times = []
test_files = sorted(glob(join(args.data_dir, "*.*")))

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

# === Runtime stats
if len(times) > 1:
    total_time = np.sum(times[1:])  # skip 1st image
    avg_time = np.mean(times[1:])
    avg_fps = 1. / avg_time

    # === Consistency check
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

    # === CSV logging
    write_header = args.overwrite_csv or not os.path.exists(args.csv_path)
    mode = 'w' if args.overwrite_csv else 'a'
    with open(args.csv_path, mode=mode, newline='') as f:
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
