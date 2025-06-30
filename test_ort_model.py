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
parser.add_argument("--model_path", type=str, default="funiegan_model_ort/funiegan_model_optimized.ort")
parser.add_argument("--csv_path", type=str, default="benchmark_results_ort.csv")
parser.add_argument("--overwrite_csv", action="store_true", help="Clear CSV before logging")
args = parser.parse_args()

# === Setup
os.makedirs(args.sample_dir, exist_ok=True)
#os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

# === ORT Runtime Session (CPU only)
sess = ort.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
print(f"Loaded model: {args.model_path}")
print(f"Input tensor name: {input_name}")

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
    img = Image.open(path).convert("RGB")
    img = img.resize((1920, 1200), Image.BICUBIC)

    img_tensor = transform(img)
    img_tensor, h, w = pad_to_multiple(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).numpy()

    start = time.time()
    output = sess.run(None, {input_name: img_tensor})[0]
    times.append(time.time() - start)

    output = torch.from_numpy(output)
    output = output[:, :, :h, :w]
    save_image(output, join(args.sample_dir, basename(path)), normalize=True)
    print(f"Processed: {basename(path)}")

# === Summary
if len(times) > 1:
    total_time = np.sum(times[1:])
    avg_time = np.mean(times[1:])
    avg_fps = 1. / avg_time
    cpu_end = psutil.cpu_percent(interval=1)
    ram_end = psutil.virtual_memory().percent

    print("\n=== ORT Model Benchmark Summary ===")
    print(f"Total images: {len(test_files)}")
    print(f"Avg time/image: {avg_time:.4f} sec")
    print(f"Avg FPS: {avg_fps:.2f}")
    print(f"[END] CPU: {cpu_end}%, RAM: {ram_end}%")
    print("===================================")

    # write_header = args.overwrite_csv or not os.path.exists(args.csv_path)
    # mode = 'w' if args.overwrite_csv else 'a'
    # with open(args.csv_path, mode=mode, newline='') as f:
    #     writer = csv.writer(f)
    #     if write_header:
    #         writer.writerow(["Model", "FPS", "TimePerImage", "CPU%", "RAM%"])
    #     writer.writerow([
    #         os.path.basename(args.model_path),
    #         round(avg_fps, 3),
    #         round(avg_time, 4),
    #         cpu_end,
    #         ram_end
    #     ])
