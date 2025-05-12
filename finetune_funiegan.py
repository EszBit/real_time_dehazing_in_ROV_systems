"""
 > Training pipeline for FUnIE-GAN (paired) model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
# py libs
import os
import glob
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
    parser.add_argument("--cfg_file", type=str, default="configs/train_target.yaml")
    #parser.add_argument("--cfg_file", type=str, default="configs/train_ufo.yaml")
    parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
    parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(torch.backends.mps.is_available())

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

    loss_G_list = []
    loss_D_list = []


    ## create dir for model and validation data
    samples_dir = os.path.join("samples_finetune/", dataset_name)
    checkpoint_dir = os.path.join("checkpoints_finetune/", dataset_name)
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
    def color_balance_loss(img):
        # img: (B, 3, H, W)
        mean_r = img[:, 0, :, :].mean()
        mean_g = img[:, 1, :, :].mean()
        mean_b = img[:, 2, :, :].mean()
        return torch.abs(mean_r - mean_g) + torch.abs(mean_r - mean_b) + torch.abs(mean_g - mean_b)

    # see if cuda is available
    if torch.cuda.is_available():
        #generator = generator.cuda()
        generator = generator.to(device)
        discriminator = discriminator.cuda()
        Adv_cGAN.cuda()
        L1_G = L1_G.cuda()
        L_vgg = L_vgg.cuda()
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    # Initialize weights or load pretrained models
    if args.epoch == 0:
        print("Loading pretrained generator...")
        generator.load_state_dict(torch.load("models/funie_generator.pth", map_location=device))
        discriminator.apply(Weights_Normal)
    else:
        generator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/generator_%d.pth" % (dataset_name, args.epoch)))
        discriminator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/discriminator_%d.pth" % (dataset_name, epoch)))
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
            imgs_distorted = Variable(batch["A"].type(Tensor))
            imgs_good_gt = Variable(batch["B"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

            ## Train Discriminator
            optimizer_D.zero_grad()
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
            optimizer_G.zero_grad()
            imgs_fake = generator(imgs_distorted)
            pred_fake = discriminator(imgs_fake, imgs_distorted)

            # GAN loss
            loss_GAN = Adv_cGAN(pred_fake, valid)

            # L1 similarity loss
            loss_1 = L1_G(imgs_fake, imgs_good_gt)

            # Perceptual VGG content loss
            loss_con = L_vgg(imgs_fake, imgs_good_gt)

            # --- 💡 Color Balance Loss ---
            # Compute the mean value per channel
            mean_channels = torch.mean(imgs_fake, dim=[2, 3])  # (batch_size, 3)
            mean_r = mean_channels[:, 0]
            mean_g = mean_channels[:, 1]
            mean_b = mean_channels[:, 2]

            # Color loss: encourages balanced RGB intensities
            loss_color = torch.mean((mean_r - mean_g) ** 2 + (mean_r - mean_b) ** 2 + (mean_g - mean_b) ** 2)

            # Combine all losses
            lambda_color = 5  # 🔧 Adjust this to control red-correction strength
            loss_G = loss_GAN + lambda_1 * loss_1 + lambda_con * loss_con + lambda_color * loss_color

            # Backward + update
            loss_G.backward()
            optimizer_G.step()

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
                imgs_val = Variable(imgs["val"].type(Tensor))
                imgs_gen = generator(imgs_val)
                img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
                save_image(img_sample, "samples_finetune/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)

        ## Save model checkpoints
        if (epoch % ckpt_interval == 0):
            torch.save(generator.state_dict(), "checkpoints_finetune/%s/generator_%d.pth" % (dataset_name, epoch))
            torch.save(discriminator.state_dict(), "checkpoints_finetune/%s/discriminator_%d.pth" % (dataset_name, epoch))

    plt.plot(loss_D_list, label="Discriminator Loss")
    plt.plot(loss_G_list, label="Generator Loss")
    plt.legend()
    plt.show()

    # === Force final save ===
    torch.save(generator.state_dict(), "checkpoints_finetune/generator_final.pth")
    torch.save(discriminator.state_dict(), "checkpoints_finetune/discriminator_final.pth")

