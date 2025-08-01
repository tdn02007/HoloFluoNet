import os
import time
import torch
import torch.nn.functional as F
import torchvision

from tqdm import tqdm
from pytorch_msssim import ssim

from data_load_torch import DatasetLoader
from network import FPN

def dice_coef(inputs, targets, smooth=1):

    inputs = F.sigmoid(inputs)
    
    intersection = (inputs * targets).sum(dim=(2, 3))
    dice = (2.*intersection + smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
    
    return dice.sum() / (inputs.size(0) * inputs.size(1))


#%%
def sample_and_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = f"checkpoint/{args.model_name}.pth"
    save_path = f"../result_mask_{args.model_name}/model_result"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+"/fake", exist_ok=True)
    os.makedirs(save_path+"/label", exist_ok=True)
    
    print("data loading")
    testset = DatasetLoader(is_train=False)

    test_loader = torch.utils.data.DataLoader(testset,
                                               batch_size=1,
                                               shuffle=False,
                                                )
    # Networks
    netG = FPN(in_channel=1, out_channel=3, is_distance=True, aspp=args.aspp, cbam=args.cbam).to(device=device)
    netV = FPN(in_channel=2, out_channel=1, is_distance=False, aspp=args.aspp, cbam=args.cbam).to(device=device)

    state_dict = torch.load(checkpoint_dir, map_location=device)

    netG.load_state_dict(state_dict["g_state_dict"])
    netV.load_state_dict(state_dict["v_state_dict"])
    
    netG.eval()
    netV.eval()
    
    with torch.no_grad():
        start = time.time()
        dice_loss_test = 0
        dice_nuclei_test = 0
        ssim_value = 0

        for i, sample in enumerate(tqdm(test_loader)):
            input_image = sample["input_image"].to(device=device, non_blocking=True)
            label_image = sample["label_image"].to(device=device, non_blocking=True)
            x3 = sample["distance_image"].to(device, non_blocking=True)

            results, h1 = netG(input_image)

            input_v = torch.cat((input_image, torch.unsqueeze(results[:, 1, :], dim=1)), axis=1)
            
            nuclei_results = netV(input_v)
            
            dice_loss_test += dice_coef(results, label_image[:, :3, :])
            dice_nuclei_test += dice_coef(nuclei_results, label_image[:, [3], :])
            ssim_value += ssim(x3, h1[:, [0], :], data_range=1)

            result_name = (4-len(str(i)))*"0" + str(i)

            pred = torch.cat(
                (
                    input_image[[0], [0], :],
                    results[[0], [0], :],
                    results[[0], [1], :],
                    results[[0], [2], :],
                    F.sigmoid(nuclei_results[[0], [0], :]),
                    h1[[0], [0], :],
                ),
                axis=-1,
            )

            torchvision.utils.save_image(
                pred,
                os.path.join(
                    save_path+"/fake",
                    f"Fake image-{result_name}.tif",
                ),
                normalize=False
            )

            label = torch.cat(
                (
                    input_image[[0], [0], :],
                    label_image[[0], [0], :],
                    label_image[[0], [1], :],
                    label_image[[0], [2], :],
                    label_image[[0], [3], :],
                    x3[[0], [0], :],
                ),
                axis=-1,
            )

            torchvision.utils.save_image(
                label,
                os.path.join(
                    save_path+"/label",
                    f"Label image-{result_name}.tif",
                ),
                normalize=False
            )
            
        total_mask_dice_loss = dice_loss_test / len(test_loader)
        total_nuclei_dice_loss = dice_nuclei_test / len(test_loader)
        total_ssim = ssim_value / len(test_loader)
        print("mask_dice_coff: ", round(total_mask_dice_loss.item(), 4))
        print("nuclei_dice_coff: ", round(total_nuclei_dice_loss.item(), 4))
        print("distance_ssim: ", round(total_ssim.item(), 4))
        print("time: ", time.time() - start)
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('HoloFluoNet parameters')

    parser.add_argument('--model_name', default='HoloFluoNet', help='name of model')

    parser.add_argument('--aspp', type=bool, default=True)
    parser.add_argument('--cbam', type=bool, default=True)

    args = parser.parse_args()

    sample_and_test(args)
    
   
                