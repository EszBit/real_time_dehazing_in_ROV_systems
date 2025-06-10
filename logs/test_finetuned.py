"""
 > Script for testing .pth models  
    * set model_name ('funiegan'/'ugan') and  model path
    * set data_dir (input) and sample_dir (output) 
"""
# py libs
import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
import psutil
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

# Options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/test/NorthWestCorner1/")
parser.add_argument("--sample_dir", type=str, default="data/output/")
parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
parser.add_argument("--model_path", type=str, default="checkpoints_finetune/generator_final.pth")
opt = parser.parse_args()

# Checks
assert exists(opt.model_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)

device = torch.device("cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Model arch
if opt.model_name.lower()=='funiegan':
    from nets import funiegan
    model = funiegan.GeneratorFunieGAN()
elif opt.model_name.lower()=='ugan':
    from nets.ugan import UGAN_Nets
    model = UGAN_Nets(base_model='pix2pix').netG
else: 
    # other models
    pass

# Load weights
model.load_state_dict(torch.load(opt.model_path, map_location=device))
model = model.to(device)
model.eval()
print ("Loaded model from %s" % (opt.model_path))

# Data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)


# Testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))

# Tracking
cpu_start = psutil.cpu_percent(interval=1)
ram_start = psutil.virtual_memory().percent
print(f"[START] CPU usage: {cpu_start}%, RAM usage: {ram_start}%")

for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = inp_img.to(device).unsqueeze(0)
    # generate enhanced image
    s = time.time()
    gen_img = model(inp_img)
    times.append(time.time()-s)
    # save output (og - enhanced)
    #img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    #save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
    # save output (only enhanced)
    save_image(gen_img.data, join(opt.sample_dir, basename(path)), normalize=True)
    print ("Tested: %s" % path)

# Run-time
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))

    cpu_end = psutil.cpu_percent(interval=1)
    ram_end = psutil.virtual_memory().percent
    print(f"[END] CPU usage: {cpu_end}%, RAM usage: {ram_end}%")

    with open("test_log.txt", "a") as f:
        f.write(f"Device: {device}, Avg FPS: {1. / Mtime:.3f}, Avg Time per image: {Mtime:.4f} sec\n")





