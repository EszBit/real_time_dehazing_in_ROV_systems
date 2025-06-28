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

import onnxruntime as ort
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch

# === Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/test/final_test/")
parser.add_argument("--sample_dir", type=str, default="data/output/final_test_results/test_onnx")
parser.add_argument("--model_path", type=str, default="models/onnx/funiegan_model_dynamic.onnx")
parser.add_argument("--device", type=str, default="cpu")  # "cpu" or "mps"
parser.add_argument("--csv_path", type=str, default="results/benchmark_results_onnx.csv")
parser.add_argument("--overwrite_csv", action="store_true", help="Clear CSV before logging")
args = parser.parse_args()

# === Setup
os.makedirs(args.sample_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

# === ONNX Runtime Session
providers = ["CPUExecutionProvider"] if args.device == "cpu" else ["CoreMLExecutionProvider"]
sess = ort.InferenceSession(args.model_path, providers=providers)
input_name = sess.get_inputs()[0].name
print(f"Using device: {args.device}, input name: {input_name}")

# === Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# === Padding
def pad_to_multiple(tensor, multiple=32):
    _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded_tensor, h, w

# === Test Loop
times = []
test_files = sorted(glob(join(args.data_dir, "*.*")))
cpu_start = psutil.cpu_percent(interval=1)
ram_start = psutil.virtual_memory().percent
print(f"[START] CPU: {cpu_start}%, RAM: {ram_start}%")

for path in test_files:
    img = Image.open(path)

    # === Resize to 1280x800 before transform
    img = img.resize((960, 600), Image.BICUBIC)

    img_tensor = transform(img)
    img_tensor, h, w = pad_to_multiple(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).numpy()  # to (1, 3, H, W) numpy

    start = time.time()
    output = sess.run(None, {input_name: img_tensor})[0]  # (1, 3, H, W)
    times.append(time.time() - start)

    # Crop back to original size
    output = torch.from_numpy(output)
    output = output[:, :, :h, :w]

    save_image(output, join(args.sample_dir, basename(path)), normalize=True)
    print(f"Tested: {path}")


# === Summary
if len(times) > 1:
    total_time = np.sum(times[1:])
    avg_time = np.mean(times[1:])
    avg_fps = 1. / avg_time
    cpu_end = psutil.cpu_percent(interval=1)
    ram_end = psutil.virtual_memory().percent

    print("\n=== ONNX Test Summary ===")
    print(f"Total images: {len(test_files)}")
    print(f"Avg time/image: {avg_time:.4f} sec")
    print(f"Avg FPS: {avg_fps:.2f}")
    print(f"[END] CPU: {cpu_end}%, RAM: {ram_end}%")
    print("====================\n")

    write_header = args.overwrite_csv or not os.path.exists(args.csv_path)
    mode = 'w' if args.overwrite_csv else 'a'
    with open(args.csv_path, mode=mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Model", "Device", "FPS", "TimePerImage", "CPU%", "RAM%"])
        writer.writerow([
            "ONNX_Model",
            args.device,
            round(avg_fps, 3),
            round(avg_time, 4),
            cpu_end,
            ram_end
        ])
