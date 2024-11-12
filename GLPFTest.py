import torch
import numpy as np
import torchvision
from torchvision import transforms
import argparse
import time
from tqdm import tqdm
from torchvision import models
from newmodelSRM_3 import *
from dataloader import UWNetDataSet
from metrics_calculation import *

__all__ = [
    "test",
    "setup",
    "testing",
]


def mask_radial(img, D0):
    # Extract dimensions from the input image tensor 'img'
    bs, ch, M, N = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    # Initialize an array to store the filter H
    H = np.zeros((bs, ch, M, N))
    # Iterate over the spatial domain to compute the filter
    for u in range(M):
        for w in range(N):
            # Calculate the distance D from the center of the image
            D = np.sqrt((u - (M / 2)) ** 2 + (w - (N / 2)) ** 2)
            H[:, :, u, w] = np.exp(-(D * D) / (2 * D0 * D0))
    return H


def convertFreqImage(img):
    # Convert the input image tensor 'img' to a numpy array
    x = img.to('cpu').detach().numpy().copy()
    bs, c, M, N = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    # Color (25) and Gray (35)
    r = 128
    H = mask_radial(np.zeros([bs, c, M, N]), r)
    H = np.real(np.fft.ifft2(H)) # to Spatial Domain Image
    TS = torch.Tensor(H)
    TS = TS.to('cuda')
    s = torch.cat((img, TS), 1) # Concat noisy image and filtered image together
    return s


@torch.no_grad()
def test(config, test_dataloader, test_model):
    test_model.eval()
    for i, (img, _, name) in enumerate(test_dataloader):
        with torch.no_grad():
            img = img.to(config.device)
            imgIR = convertFreqImage(img)
            generate_img = test_model(imgIR)
            torchvision.utils.save_image(generate_img, config.output_images_path + name[0])


def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    model = torch.load(config.snapshot_path).to(config.device)
    transform = transforms.Compose([transforms.Resize((config.resize, config.resize)), transforms.ToTensor()])
    test_dataset = UWNetDataSet(config.test_images_path, None, transform, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print("Test Dataset Reading Completed.")
    return test_dataloader, model


def testing(config):
    ds_test, model = setup(config)
    test(config, ds_test, model)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Trainable Parameters", pytorch_total_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot_path', type=str, default='./IR_GLPF'
                                                             '/model_epoch_48.ckpt',
                        help='snapshot path,such as :xxx/snapshots/model.ckpt default:None')
    parser.add_argument('--test_images_path', type=str, default="./data/EU_Dark/input/",
                        help='path of input images(underwater images) for te8ting default:./data/input/')
    parser.add_argument('--output_images_path', type=str,
                        default='./results/EU_Dark_GLPF/',
                        help='path to save generated image.')
    parser.add_argument('--batch_size', type=int, default=1, help="default : 1")
    parser.add_argument('--resize', type=int, default=256, help="resize images, default:resize images to 256*256")

    parser.add_argument('--calculate_metrics', type=bool, default=True,
                        help="calculate PSNR, SSIM and UIQM on test images")
    parser.add_argument('--label_images_path', type=str, default="./data/EU_Dark/label/",
                        help='path of label images(clear images) default:./data/laEUP-Dbel/')

    print("-------------------testing---------------------")
    config = parser.parse_args()
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)

    start_time = time.time()
    testing(config)

    print("total testing time", time.time() - start_time)

    if config.calculate_metrics:
        print("-------------------calculating performance metrics---------------------")
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path, config.label_images_path,
                                                                   (config.resize, config.resize))
        UIQM_measures = calculate_UIQM(config.output_images_path, (config.resize, config.resize))

        print("SSIM on {0} samples {1} ± {2}".format(len(SSIM_measures), np.round(np.mean(SSIM_measures), 3),
                                                     np.round(np.std(SSIM_measures), 3)))
        print("PSNR on {0} samples {1} ± {2}".format(len(PSNR_measures), np.round(np.mean(PSNR_measures), 3),
                                                     np.round(np.std(PSNR_measures), 3)))
        print("UIQM on {0} samples {1} ± {2}".format(len(UIQM_measures), np.round(np.mean(UIQM_measures), 3),
                                                     np.round(np.std(UIQM_measures), 3)))
