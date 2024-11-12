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
    "Trainer",  # Trainer class
    "setup",  # setup function
    "training",  # training module
]


def mask_radial(img, D0, n):
    # Get the dimensions of the input image
    bs, ch, M, N = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    H = np.zeros((bs, ch, M, N))
    rows, columns = img.shape[2], img.shape[3]
    # Calculate the midpoint of rows and columns
    mid_R, mid_C = int(rows / 2), int(columns / 2)
    for i in range(rows):
        for j in range(columns):
            d = np.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
            H[:, :, i, j] = 1 / (1 + (d / D0) ** (2 * n))
    return H


def convertFreqImage(img):  # Convert Frequency Domain
    bs, c, M, N = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    r = 128 # The image has a size of 256, and the threshold value, set at pi/2 [128]
    n = 1
    H = mask_radial(np.zeros([bs, c, M, N]), r, n)
    # Apply inverse Fourier Transform to obtain spatial domain representation
    H = np.real(np.fft.ifft2(H))
    TS = torch.Tensor(H)
    TS = TS.to('cuda')
    # Concatenate the noisy image and the spatial domain filtered image
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
        # Get the device configuration from the provided config dictionary
        device = config['device']
        # Initialize lists to store losses during training
        primary_loss_lst = []
        vgg_loss_lst = []
        total_loss_lst = []
        # Evaluate the model on the test dataset
        UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
        # Log evaluation metrics to Weights & Biases (wandb)
        wandb.log({f"[Test] Epoch": 0,
                   "[Test] UIQM": np.mean(UIQM),
                   "[Test] SSIM": np.mean(SSIM),
                   "[Test] PSNR": np.mean(PSNR), },
                  commit=True
                  )

        for epoch in trange(0, config.num_epochs, desc=f"[Full Loop]", leave=False):
            # Initialize temporary variables to store losses for this epoch
            primary_loss_tmp = 0
            vgg_loss_tmp = 0
            total_loss_tmp = 0
            # Adjust learning rate if applicable
            if epoch > 1 and epoch % config.step_size == 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.7
            # Iterate over the training dataloader for this epoch
            for inp, label, _ in tqdm(train_dataloader, desc=f"[Train]", leave=False):
                inp = inp.to(device)
                inpIR = convertFreqImage(inp)
                label = label.to(device)
                # Set the model to training mode and zero the gradients
                self.model.train()
                self.opt.zero_grad()
                # Forward pass
                out = self.model(inpIR)
                # Calculate loss and perform backpropagation
                loss, mse_loss, vgg_loss = self.loss(out, label)
                loss.backward()
                self.opt.step()
                # Accumulate losses for this batch
                primary_loss_tmp += mse_loss.item()
                vgg_loss_tmp += vgg_loss.item()
                total_loss_tmp += loss.item()
            # Calculate average losses for this epoch and log to Weights & Biases
            total_loss_lst.append(total_loss_tmp / len(train_dataloader))
            vgg_loss_lst.append(vgg_loss_tmp / len(train_dataloader))
            primary_loss_lst.append(primary_loss_tmp / len(train_dataloader))
            wandb.log({f"[Train] Total Loss": total_loss_lst[epoch],
                       "[Train] Primary Loss": primary_loss_lst[epoch],
                       "[Train] VGG Loss": vgg_loss_lst[epoch], },
                      commit=True
                      )
            # Evaluate the model on the test dataset
            if (config.test == True) & (epoch % config.eval_steps == 0):
                UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
                wandb.log({f"[Test] Epoch": epoch + 1,
                           "[Test] UIQM": np.mean(UIQM),
                           "[Test] SSIM": np.mean(SSIM),
                           "[Test] PSNR": np.mean(PSNR), },
                          commit=True
                          )
            # Print training progress
            if epoch % config.print_freq == 0:
                print('epoch:[{}]/[{}], image loss:{}, MSE / L1 loss:{}, VGG loss:{}'.format(epoch, config.num_epochs,
                                                                                             str(total_loss_lst[epoch]),
                                                                                             str(primary_loss_lst[
                                                                                                     epoch]),
                                                                                             str(vgg_loss_lst[epoch])))
            # Save model checkpoints
            if not os.path.exists(config.snapshots_folder):
                os.mkdir(config.snapshots_folder)

            if epoch % config.snapshot_freq == 0:
                torch.save(self.model, config.snapshots_folder + 'model_epoch_{}.ckpt'.format(epoch))

    @torch.no_grad()
    # Define a function to evaluate the model
    def eval(self, config, test_dataloader, test_model):
        # Set the model to evaluation mode
        test_model.eval()
        for i, (img, _, name) in enumerate(test_dataloader):
            with torch.no_grad():
                # Move input data to the specified device (cuda or cpu)
                img = img.to(config.device)
                # Concat with noisy image and filtered image together
                imgIR = convertFreqImage(img)
                generate_img = test_model(imgIR)
                # Save generated image
                torchvision.utils.save_image(generate_img, config.output_images_path + name[0])
        # Calculate SSIM and PSNR metrics
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path,
                                                                   config.GTr_test_images_path)
        # Calculate UIQM metric
        UIQM_measures = calculate_UIQM(config.output_images_path)
        return UIQM_measures, SSIM_measures, PSNR_measures


def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
    # Initialize the neural network model with a specified number of layers
    model = UWnet(num_layers=config.num_layers).to(config["device"])
    # Resize the images to a specified size
    transform = transforms.Compose([transforms.Resize((config.resize, config.resize)), transforms.ToTensor()])
    # Create a data loader for the training dataset
    train_dataset = UWNetDataSet(config.input_images_path, config.label_images_path, transform, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False)
    print("Train Dataset Reading Completed.")

    loss = combinedloss(config)
    # Initialize the optimizer with the model parameters and learning rate
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    # Create a trainer object for training the model
    trainer = Trainer(model, opt, loss)

    if config.test:
        # Create a dataset for testing using the provided test images path
        test_dataset = UWNetDataSet(config.test_images_path, None, transform, False)
        # Create a data loader for the test dataset with specified batch size
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
        print("Test Dataset Reading Completed.")
        return train_dataloader, test_dataloader, model, trainer
    return train_dataloader, None, model, trainer


def training(config):
    # Initialize wandb with the specified project
    wandb.init(project="underwater_image_enhancement_UWNet")
    # Update wandb configuration with the provided configuration (config) and allow validation change
    wandb.config.update(config, allow_val_change=True)
    # Retrieve the updated configuration from wandb
    config = wandb.config
    # Setup training and testing datasets, model, and trainer based on the updated configuration
    ds_train, ds_test, model, trainer = setup(config)
    # Train the model using the training dataset (ds_train)
    # and evaluate on the testing dataset (ds_test)
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
    parser.add_argument('--snapshots_folder', type=str, default="./IR_BLPF/")
    parser.add_argument('--output_images_path', type=str, default="./ValImg"
                                                                  "/IR_BLPF/")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)

    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)
    training(config)
