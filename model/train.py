import os
import sys
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from pytorch_msssim import ssim

from network import FPN, Discriminator
from loss_torch import InclusionLoss, ExclusionLoss, GANLoss, DiceCELoss
from data_load_torch import DatasetLoader

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)



def dice_coef(inputs, targets, smooth=1.0):

    inputs = F.sigmoid(inputs)

    intersection = (inputs * targets).sum(dim=(2, 3))
    dice = (2.*intersection + smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)

    return dice.sum() / (inputs.size(0) * inputs.size(1))


def train(args):
    print("data loading")
    batch_size = args.batch_size
    
    dataset = DatasetLoader(is_train=True)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )

    testset = DatasetLoader(is_train=False)
    test_loader = torch.utils.data.DataLoader(testset,
                                               batch_size=1,
                                               shuffle=True,
                                                )

    # Networks
    netG = FPN(in_channel=1, out_channel=3, is_distance=True, aspp=args.aspp, cbam=args.cbam).to(device=device)
    netV = FPN(in_channel=2, out_channel=1, is_distance=False, aspp=args.aspp, cbam=args.cbam).to(device=device)

    netD1 = Discriminator(in_channel=4).to(device=device)
    netD2 = Discriminator(in_channel=2).to(device=device)

    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)

    optimizerV = optim.Adam(netV.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    schedulerV = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerV, args.num_epoch, eta_min=1e-5)

    optimizerD1 = optim.Adam(netD1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    schedulerD1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD1, args.num_epoch, eta_min=1e-5)

    optimizerD2 = optim.Adam(netD2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    schedulerD2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD2, args.num_epoch, eta_min=1e-5)


    dice_ce_loss = DiceCELoss().to(device)
    inclusion_loss = InclusionLoss().to(device)
    exclusion_loss = ExclusionLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionGAN = GANLoss().to(device)

    last_dice_loss = 0

    for epoch in range(args.num_epoch):
        print("epoch: ", epoch)
        G_loss = 0
        D1_loss = 0
        D2_loss = 0

        for sample in tqdm(train_loader):
            x1 = sample["input_image"].to(device, non_blocking=True)
            x2 = sample["label_image"].to(device, non_blocking=True)
            x3 = sample["distance_image"].to(device, non_blocking=True)

            # updated D1
            for p in netD1.parameters():
                p.requires_grad = True

            optimizerD1.zero_grad()

            results, h1 = netG(x1)

            pred_fake = netD1(x1, results)

            loss_D1_fake = criterionGAN(pred_fake, False)

            pred_real = netD1(x1, x2[:, :3, :])
            loss_D1_real = criterionGAN(pred_real, True)

            loss_D1 = (loss_D1_fake + loss_D1_real) * 0.5

            D1_loss += loss_D1.item()
            loss_D1.backward(retain_graph=True)
            optimizerD1.step()

            # updated D2
            for p in netD2.parameters():
                p.requires_grad = True

            optimizerD2.zero_grad()

            results, h1 = netG(x1)
            input_v = torch.cat((x1, results[:, [1], :]), axis=1)
            nuclei_results = netV(input_v)

            pred_fake = netD2(x1, nuclei_results)

            loss_D2_fake = criterionGAN(pred_fake, False)

            pred_real = netD2(x1, x2[:, [3], :])
            loss_D2_real = criterionGAN(pred_real, True)

            loss_D2 = (loss_D2_fake + loss_D2_real) * 0.5

            D2_loss += loss_D2.item()
            loss_D2.backward()
            optimizerD2.step()

            #update G
            for p in netD1.parameters():
                p.requires_grad = False
            for p in netD2.parameters():
                p.requires_grad = False

            optimizerG.zero_grad()
            optimizerV.zero_grad()

            results, h1 = netG(x1)
            pred_real_D1 = netD1(x1, results)
            
            errG1_Dice = dice_ce_loss(results, x2[:, :3, :])
            if args.inclusion:
                errG1_inclusion = inclusion_loss(results[:, 1, :], x2[:, 3, :])
            else:
                errG1_inclusion = torch.tensor(0.0, device=device)
            if args.exclusion:
                errG1_exclusion = exclusion_loss(results[:, 2, :], x2[:, 3, :])
            else:
                errG1_exclusion = torch.tensor(0.0, device=device)
            errG1_L1_distance = criterionL1(h1, x3)
            errG1_gan_1 = criterionGAN(pred_real_D1, True)
            
            input_v = torch.cat((x1, results[:, [1], :]), axis=1)

            nuclei_results = netV(input_v)
            pred_real_D2 = netD2(x1, nuclei_results)

            errV1_Dice = dice_ce_loss(nuclei_results, x2[:, [3], :])
            errG1_gan_2 = criterionGAN(pred_real_D2, True)

            errG = errG1_Dice + errG1_inclusion + errG1_exclusion + errV1_Dice + errG1_L1_distance + errG1_gan_1 + errG1_gan_2
            
            G_loss += errG
            
            errG.backward()
            optimizerG.step()
            optimizerV.step()

        print('epoch {}, G Loss: {}'.format(epoch, G_loss.item() / len(train_loader)))

        if not args.no_lr_decay:
            schedulerG.step()
            schedulerV.step()
            schedulerD1.step()
            schedulerD2.step()

        dice_mask_test = 0
        dice_nuclei_test = 0
        ssim_distance_test = 0

        with torch.no_grad():
            print("Testing")
            for i, sample in enumerate(test_loader):
                x1 = sample["input_image"].to(device, non_blocking=True)
                x2 = sample["label_image"].to(device, non_blocking=True)
                x3 = sample["distance_image"].to(device, non_blocking=True)

                results, h1 = netG(x1)

                input_v = torch.cat((x1, torch.unsqueeze(results[:, 1, :], dim=1)), axis=1)
                
                nuclei_results = netV(input_v)

                dice_mask_test += dice_coef(results, x2[:, :3, :])
                dice_nuclei_test += dice_coef(nuclei_results, x2[:, [3], :])
                ssim_distance_test += ssim(x3, h1[:, [0], :], data_range=1)


            total_mask_dice_loss = dice_mask_test / len(test_loader)
            total_nuclei_dice_loss = dice_nuclei_test / len(test_loader)
            total_distance_ssim_loss = ssim_distance_test / len(test_loader)
            print("dice_mask_test: ", round(total_mask_dice_loss.item(), 4))
            print("dice_nuclei_test: ", round(total_nuclei_dice_loss.item(), 4))
            print("ssim_distance_test: ", round(total_distance_ssim_loss.item(), 4))

            total_dice_loss = (total_mask_dice_loss + total_nuclei_dice_loss)/2

            if last_dice_loss < total_dice_loss:
                print("Saving model...")

                model_out_path = f"checkpoint/{args.model_name}.pth"
                os.makedirs("checkpoint", exist_ok=True)

                torch.save(
                    {
                        "g_state_dict": netG.state_dict(),
                        "v_state_dict": netV.state_dict(),
                    },
                    model_out_path,
                )

                last_dice_loss = total_dice_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HoloFluoNet parameters')

    parser.add_argument('--model_name', default='HoloFluoNet', help='name of model')

    parser.add_argument('--aspp', type=bool, default=True)
    parser.add_argument('--cbam', type=bool, default=True)
    parser.add_argument('--inclusion', type=bool, default=True)
    parser.add_argument('--exclusion', type=bool, default=True)

    parser.add_argument('--batch_size', type=int, default=24, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=100)

    parser.add_argument('--lr_g', type=float, default=0.001, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    args = parser.parse_args()
    train(args)