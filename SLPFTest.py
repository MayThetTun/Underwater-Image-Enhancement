import torchvision
from torchvision import transforms
import argparse
import time
from newmodelSRM_3 import *
from dataloader import UWNetDataSet
from metrics_calculation import *

__all__ = [
    "test",
    "setup",
    "testing",
]


def PowerSpectrum(x):
    # Extract dimensions from the input numpy array 'x'
    M, N = x.shape[0], x.shape[1]
    # Calculate the midpoints for M and N
    MS = M // 2
    NS = N // 2
    # Compute the Fourier transform of the input array 'x'
    F = np.fft.fftn(x)
    # Shift the zero frequency component to the center of the spectrum
    F_magnitude = np.fft.fftshift(F)
    # Compute the power spectrum
    PS = (np.abs(F_magnitude) ** 2)
    # Extract horizontal and vertical frequencies at the center of the power spectrum
    Htt = PS[MS,]
    Vtt = PS[:, MS]
    # Compute the sum of the horizontal and vertical frequency
    SS = np.sum(np.sum(np.ravel(Htt)) + np.sum(np.ravel(Vtt)))
    # Calculate power spectrum sparsity
    Spar = np.sum(np.ravel(PS)) / SS
    # Threshold calculation
    Thre = Spar * 256
    # Convert threshold value to integer
    K = int(Thre)
    PS[MS - K: MS + K, NS - K: NS + K] = 0
    # Specify the frequency response of Sparsity based LPF
    peaks = PS <= Thre
    # Inverse Fourier transform and shift back to the original position
    iffts = np.fft.ifftshift(peaks).astype(int)
    # Apply inverse Fourier transform to obtain the spatial domain filtered images
    image_filtered = np.fft.ifft2(iffts)
    return image_filtered


def convertFreqImage(img):  # Convert Frequency Domain
    x = img.to('cpu').detach().numpy().copy()
    bs, c, M, N = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    S = np.zeros((bs, c, M, N), dtype=np.float32)
    for bs in range(bs):
        for ch in range(c):
            S[bs, ch, :, :] = PowerSpectrum(x[bs, ch, :, :])
    TS = torch.Tensor(S)
    TS = TS.cuda()
    s = torch.cat((img, TS), 1) # concat with noisy image and filtered image
    return s


@torch.no_grad()
def test(config, test_dataloader, test_model):
    test_model.eval()
    for i, (img, _, name) in enumerate(test_dataloader):
        with torch.no_grad():
            img = img.to(config.device)
            imgIR = convertFreqImage(img) # Concat with filtered images
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
    parser.add_argument('--snapshot_path', type=str, default='./IR_SLPF'
                                                             '/model_epoch_48.ckpt',
                        help='snapshot path,such as :xxx/snapshots/model.ckpt default:None')
    parser.add_argument('--test_images_path', type=str, default="./data/EU_Dark/input/",
                        help='path of input images(underwater images) for te8ting default:./data/UFO/input/')
    parser.add_argument('--output_images_path', type=str,
                        default='./results/EU_Dark_SLPF/',
                        help='path to save generated image.')
    parser.add_argument('--batch_size', type=int, default=1, help="default : 1")
    parser.add_argument('--resize', type=int, default=256, help="resize images, default:resize images to 256*256")

    parser.add_argument('--calculate_metrics', type=bool, default=True,
                        help="calculate PSNR, SSIM and UIQM on test images")
    parser.add_argument('--label_images_path', type=str, default="./data/EU_Dark/label/",
                        help='path of label images(clear images) default:./data/UFO/label/')

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
