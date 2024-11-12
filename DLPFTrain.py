from torch.nn import Module
import torchvision
from torchvision import transforms
import wandb
import argparse
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
from dataloader import UWNetDataSet
from metrics_calculation import *
from newmodelSRM_3 import *
from combined_loss import *

__all__ = [
    "Trainer",
    "setup",
    "training",
]


def distance(i, j, imageSize, r):
    # Calculate the Euclidean distance of the pixel from the center of the image
    dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
    # Check if the distance is less than or equal to the specified radius 'r'
    if dis <= r:
        return 1.0
    else:
        return 0


def mask_radial(img, r):
    # Extract dimensions of the input image tensor
    bs, ch, rows, cols = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    # Initialize an array of zeros to store the mask with the same shape as the input image
    mask = np.zeros((bs, ch, rows, cols))
    # Loop through each pixel in the image
    for i in range(rows):
        for j in range(cols):
            # Calculate the distance of the current pixel from the center of the image
            mask[:, :, i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def convertFreqImage(img):
    # Convert the image tensor to a numpy array
    x = img.to('cpu').detach().numpy().copy()
    # Extract dimensions of the input numpy array
    bs, c, M, N = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    r = 128  # The image has a size of 256, and the threshold value, set at pi/2 [128]
    # Generate a radial mask using a function 'mask_radial
    H = mask_radial(np.zeros([bs, c, M, N]), r)
    # To convert to spatial domain image
    H = np.fft.ifft2(H)
    TS = torch.Tensor(H)
    TS = TS.to('cuda')
    # Concat with noisy image and filtered image
    s = torch.cat((img, TS), 1)
    return s


## TODO: Update config parameter names
## TODO: remove wandb steps
## TODO: Add comments to functions

@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module

    @torch.enable_grad()
    def train(self, train_dataloader, config, test_dataloader=None):
        device = config['device']
        primary_loss_lst = []
        vgg_loss_lst = []
        total_loss_lst = []
        # Evaluate the model on the test dataset and retrieve evaluation metrics
        UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
        # Log evaluation metrics using Weights & Biases (wandb)
        wandb.log({f"[Test] Epoch": 0,
                   "[Test] UIQM": np.mean(UIQM),
                   "[Test] SSIM": np.mean(SSIM),
                   "[Test] PSNR": np.mean(PSNR), },
                  commit=True
                  )
        # Iterate over epochs using tqdm for progress visualization
        for epoch in trange(0, config.num_epochs, desc=f"[Full Loop]", leave=False):
            primary_loss_tmp = 0
            vgg_loss_tmp = 0
            total_loss_tmp = 0
            # Learning rate decay scheduler
            if epoch > 1 and epoch % config.step_size == 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.7

            for inp, label, _ in tqdm(train_dataloader, desc=f"[Train]", leave=False):
                inp = inp.to(device)
                inpIR = convertFreqImage(inp)
                label = label.to(device)
                self.model.train()
                self.opt.zero_grad()
                out = self.model(inpIR)  # Forward pass
                loss, mse_loss, vgg_loss = self.loss(out, label)  # Calculate loss
                loss.backward()  # Backpropagation
                self.opt.step()  # Update model parameters
                primary_loss_tmp += mse_loss.item()  # Accumulate primary loss
                vgg_loss_tmp += vgg_loss.item()  # Accumulate VGG loss
                total_loss_tmp += loss.item()  # Accumulate total loss
            # Calculate average losses and store them
            total_loss_lst.append(total_loss_tmp / len(train_dataloader))
            vgg_loss_lst.append(vgg_loss_tmp / len(train_dataloader))
            primary_loss_lst.append(primary_loss_tmp / len(train_dataloader))
            # Log training losses
            wandb.log({f"[Train] Total Loss": total_loss_lst[epoch],
                       "[Train] Primary Loss": primary_loss_lst[epoch],
                       "[Train] VGG Loss": vgg_loss_lst[epoch], },
                      commit=True
                      )
            # Evaluate model on test dataset if test mode is enabled
            if (config.test == True) & (epoch % config.eval_steps == 0):
                UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
                wandb.log({f"[Test] Epoch": epoch + 1,
                           "[Test] UIQM": np.mean(UIQM),
                           "[Test] SSIM": np.mean(SSIM),
                           "[Test] PSNR": np.mean(PSNR), },
                          commit=True
                          )

            if epoch % config.print_freq == 0:
                print('epoch:[{}]/[{}], image loss:{}, MSE / L1 loss:{}, VGG loss:{}'.format(epoch, config.num_epochs,
                                                                                             str(total_loss_lst[epoch]),
                                                                                             str(primary_loss_lst[
                                                                                                     epoch]),
                                                                                             str(vgg_loss_lst[epoch])))

            # Save model snapshot
            if not os.path.exists(config.snapshots_folder):
                os.mkdir(config.snapshots_folder)

            if epoch % config.snapshot_freq == 0:
                torch.save(self.model, config.snapshots_folder + 'model_epoch_{}.ckpt'.format(epoch))

    @torch.no_grad()
    def eval(self, config, test_dataloader, test_model):
        # Set the model to evaluation mode
        test_model.eval()
        for i, (img, _, name) in enumerate(test_dataloader):
            with torch.no_grad():   # Disable gradient calculation
                img = img.to(config.device)
                imgIR = convertFreqImage(img)
                generate_img = test_model(imgIR)
                torchvision.utils.save_image(generate_img, config.output_images_path + name[0])
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path,
                                                                   config.GTr_test_images_path)
        UIQM_measures = calculate_UIQM(config.output_images_path)
        return UIQM_measures, SSIM_measures, PSNR_measures


def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    model = UWnet(num_layers=config.num_layers).to(config["device"])
    # Define transformations for input images
    transform = transforms.Compose([transforms.Resize((config.resize, config.resize)), transforms.ToTensor()])
    # Create training dataset
    train_dataset = UWNetDataSet(config.input_images_path, config.label_images_path, transform, True)
    # Create dataloader for training dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False)
    print("Train Dataset Reading Completed.")
    # Define the loss function
    loss = combinedloss(config)
    # Define the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    # Create trainer object for training the model
    trainer = Trainer(model, opt, loss)

    if config.test:
        # Create test dataset
        test_dataset = UWNetDataSet(config.test_images_path, None, transform, False)
        # Create dataloader for test dataset
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
        print("Test Dataset Reading Completed.")
        return train_dataloader, test_dataloader, model, trainer
    return train_dataloader, None, model, trainer


def training(config):
    # Initialize wandb project
    wandb.init(project="underwater_image_enhancement_UWNet")
    # Update wandb config with the current configuration settings
    wandb.config.update(config, allow_val_change=True)
    config = wandb.config
    # Setup training and optionally testing datasets, model, and trainer
    ds_train, ds_test, model, trainer = setup(config)
    trainer.train(ds_train, config, ds_test)
    print("==================")
    print("Training complete!")
    print("==================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--input_images_path', type=str, default="./data/EUVP/train/input/",
                        help='path of input images(underwater images) default:./data/input/')
    parser.add_argument('--label_images_path', type=str, default="./data/EUVP/train/label/",
                        help='path of label images(clear images) default:./data/label/')
    parser.add_argument('--test_images_path', type=str, default="./data/EUVP/Val/input/",
                        help='path of input images(underwater images) for testing default:./data/input/')
    parser.add_argument('--GTr_test_images_path', type=str, default="./data/EUVP/Val/label/",
                        help='path of input ground truth images(underwater images) for testing default:./data/input/')
    parser.add_argument('--test', default=True)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--step_size', type=int, default=400, help="Period of learning rate decay")  # 50
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=1, help="default : 1")
    parser.add_argument('--test_batch_size', type=int, default=1, help="default : 1")
    parser.add_argument('--resize', type=int, default=256, help="resize images, default:resize images to 256*256")
    parser.add_argument('--cuda_id', type=int, default=0, help="id of cuda device,default:0")
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--snapshot_freq', type=int, default=2)
    parser.add_argument('--snapshots_folder', type=str, default="./IR_DLPF/")
    parser.add_argument('--output_images_path', type=str, default="./ValImg"
                                                                  "/IR_DLPF/")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)
    # Parse command line arguments and store them in config
    config = parser.parse_args()
    # Check if the snapshots folder path specified in the config exists, if not, create it
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    # Check if the output images folder path specified in the config exists, if not, create it
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)
    # Start training process with the provided configuration
    training(config)
