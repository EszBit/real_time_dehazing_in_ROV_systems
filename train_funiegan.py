"""
 > Training pipeline for FUnIE-GAN (paired) model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
# py libs
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from utils.data_utils import GetTrainingPairs, GetValImage
import matplotlib.pyplot as plt


#print(torch.backends.mps.is_available())
#print(torch.backends.mps.is_built())


if __name__ == "__main__":

    ## get configs and training options
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/train_euvp.yaml")
    #parser.add_argument("--cfg_file", type=str, default="configs/train_ufo.yaml")
    parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
    parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    # learning rate was 0.0003, changed to 0.0001
    parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    ## training params
    epoch = args.epoch
    num_epochs = args.num_epochs
    batch_size =  args.batch_size
    lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2
    # load the data config file
    with open(args.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # get info from config file
    dataset_name = cfg["dataset_name"]
    dataset_path = cfg["dataset_path"]
    channels = cfg["chans"]
    img_width = cfg["im_width"]
    img_height = cfg["im_height"]
    val_interval = cfg["val_interval"]
    ckpt_interval = cfg["ckpt_interval"]

    # Loss function plot
    loss_D_list = []
    loss_G_list = []

    ## create dir for model and validation data
    samples_dir = os.path.join("samples/FunieGAN/", dataset_name)
    checkpoint_dir = os.path.join("checkpoints/", dataset_name)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


    """ FunieGAN specifics: loss functions and patch-size
    -----------------------------------------------------"""
    Adv_cGAN = torch.nn.MSELoss()
    L1_G  = torch.nn.L1Loss() # similarity loss (l1)
    L_vgg = VGG19_PercepLoss() # content loss (vgg)
    lambda_1, lambda_con = 7, 3 # 7:3 (as in paper)
    patch = (1, img_height//16, img_width//16) # 16x16 for 256x256

    # Initialize generator and discriminator
    generator = GeneratorFunieGAN()
    discriminator = DiscriminatorFunieGAN()

    # Color balance loss function
    #def color_balance_loss(img):
        # img: (B, 3, H, W)
        #mean_r = img[:, 0, :, :].mean()
        #mean_g = img[:, 1, :, :].mean()
        #mean_b = img[:, 2, :, :].mean()
        #return torch.abs(mean_r - mean_g) + torch.abs(mean_r - mean_b) + torch.abs(mean_g - mean_b)

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    Adv_cGAN = Adv_cGAN.to(device)
    L1_G = L1_G.to(device)
    L_vgg = L_vgg.to(device)

    # Initialize weights or load pretrained models
    if args.epoch == 0:
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)
    else:
        generator.load_state_dict(torch.load("checkpoints/%s/generator_%d.pth" % (dataset_name, args.epoch)))
        discriminator.load_state_dict(torch.load("checkpoints/%s/discriminator_%d.pth" % (dataset_name, epoch)))
        print ("Loaded model from epoch %d" %(epoch))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))


    ## Data pipeline
    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
        batch_size = batch_size,
        shuffle = True,
        num_workers = 8,
    )

    val_dataloader = DataLoader(
        GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
        batch_size=4,
        shuffle=True,
        num_workers=1,
    )


    ## Training pipeline
    for epoch in range(epoch, num_epochs):
        for i, batch in enumerate(dataloader):
            # Model inputs
            imgs_distorted = batch["A"].to(device)
            imgs_good_gt = batch["B"].to(device)
            # Adversarial ground truths
            valid = torch.ones((imgs_distorted.size(0), *patch), device=device, dtype=torch.float32,
                               requires_grad=False)
            fake = torch.zeros((imgs_distorted.size(0), *patch), device=device, dtype=torch.float32,
                               requires_grad=False)

            ## Train Discriminator
            optimizer_D.zero_grad() # clears prev. computed gradients
            imgs_fake = generator(imgs_distorted)
            pred_real = discriminator(imgs_good_gt, imgs_distorted)
            loss_real = Adv_cGAN(pred_real, valid)
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_fake = Adv_cGAN(pred_fake, fake)
            # Total loss: real + fake (standard PatchGAN)
            loss_D = 0.5 * (loss_real + loss_fake) * 10.0 # 10x scaled for stability
            loss_D.backward()
            optimizer_D.step()

            ## Train Generator
            optimizer_G.zero_grad() # clears prev. computed gradients
            imgs_fake = generator(imgs_distorted)
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_GAN =  Adv_cGAN(pred_fake, valid) # GAN loss
            loss_1 = L1_G(imgs_fake, imgs_good_gt) # similarity loss
            loss_con = L_vgg(imgs_fake, imgs_good_gt)# content loss

            # New color balance loss
            #loss_color = color_balance_loss(imgs_fake)

            # Update total generator loss (adding color balance loss with a small weight)
            lambda_color = 1.5
            #loss_G = loss_GAN + lambda_1 * loss_1 + lambda_con * loss_con + lambda_color * loss_color
            loss_G = loss_GAN + lambda_1 * loss_1 + lambda_con * loss_con
            # Total loss (Section 3.2.1 in the paper)
            # loss_G = loss_GAN + lambda_1 * loss_1  + lambda_con * loss_con
            loss_G.backward()
            optimizer_G.step()

            # Loss function plot
            loss_D_list.append(loss_D.item())
            loss_G_list.append(loss_G.item())

            ## Print log
            if not i%50:
                sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f]"
                                  %(
                                    epoch, num_epochs, i, len(dataloader),
                                    loss_D.item(), loss_G.item(), loss_GAN.item(),
                                   )
                )
            ## If at sample interval save image
            batches_done = epoch * len(dataloader) + i
            if batches_done % val_interval == 0:
                imgs = next(iter(val_dataloader))
                imgs_val = imgs["val"].to(device)
                imgs_gen = generator(imgs_val)
                img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
                save_image(img_sample, "samples/FunieGAN/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)

        # Save model checkpoints every `ckpt_interval` epochs and at the final epoch
        if (epoch % ckpt_interval == 0 or epoch == num_epochs - 1):
            torch.save(generator.state_dict(), f"checkpoints/FunieGAN/{dataset_name}/generator_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/FunieGAN/{dataset_name}/discriminator_{epoch}.pth")

    # Loss function plot
    plt.plot(loss_D_list, label="Discriminator Loss")
    plt.plot(loss_G_list, label="Generator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === Force final save ===
    torch.save(generator.state_dict(), "checkpoints/FunieGAN/EUVP/generator_final.pth")
    torch.save(discriminator.state_dict(), "checkpoints/FunieGAN/EUVP/discriminator_final.pth")

