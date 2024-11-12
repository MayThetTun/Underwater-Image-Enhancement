from torch.nn import Module
import torchvision
from torchvision import transforms
import wandb
import argparse
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
from dataloader import UWNetDataSet
from metrics_calculation import *
from model import *
from combined_loss import *

__all__ = [
    "Trainer",
    "setup",
    "training",
]

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
        # Evaluate the model on the test dataset
        UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
        # Log the evaluation results using Weights & Biases (wandb)
        wandb.log({f"[Test] Epoch": 0,
                   "[Test] UIQM": np.mean(UIQM),
                   "[Test] SSIM": np.mean(SSIM),
                   "[Test] PSNR": np.mean(PSNR), },
                  commit=True
                  )
        # Iterate through epochs
        for epoch in trange(0, config.num_epochs, desc=f"[Full Loop]", leave=False):
            # Initialize temporary variables for losses
            primary_loss_tmp = 0
            vgg_loss_tmp = 0
            total_loss_tmp = 0
            # Adjust learning rate every 'step_size' epochs
            if epoch > 1 and epoch % config.step_size == 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.7
            # Iterate through the training dataloader
            for inp, label, _ in tqdm(train_dataloader, desc=f"[Train]", leave=False):
                inp = inp.to(device)
                label = label.to(device)
                # Set the model to training mode
                self.model.train()
                # Reset gradients
                self.opt.zero_grad()
                # Forward pass
                out = self.model(inp)
                # Compute loss
                loss, mse_loss, vgg_loss = self.loss(out, label)
                # Backpropagation
                loss.backward()
                self.opt.step()
                # Update temporary loss variables
                primary_loss_tmp += mse_loss.item()
                vgg_loss_tmp += vgg_loss.item()
                total_loss_tmp += loss.item()
            # Calculate average losses for the epoch
            total_loss_lst.append(total_loss_tmp / len(train_dataloader))
            vgg_loss_lst.append(vgg_loss_tmp / len(train_dataloader))
            primary_loss_lst.append(primary_loss_tmp / len(train_dataloader))
            # Log training losses using Weights & Biases (wandb)
            wandb.log({f"[Train] Total Loss": total_loss_lst[epoch],
                       "[Train] Primary Loss": primary_loss_lst[epoch],
                       "[Train] VGG Loss": vgg_loss_lst[epoch], },
                      commit=True
                      )
            # Evaluate on test dataset if specified and at 'eval_steps' epochs
            if (config.test == True) & (epoch % config.eval_steps == 0):
                UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
                wandb.log({f"[Test] Epoch": epoch + 1,
                           "[Test] UIQM": np.mean(UIQM),
                           "[Test] SSIM": np.mean(SSIM),
                           "[Test] PSNR": np.mean(PSNR), },
                          commit=True
                          )
            # Print progress every 'print_freq' epochs
            if epoch % config.print_freq == 0:
                print('epoch:[{}]/[{}], image loss:{}, MSE / L1 loss:{}, VGG loss:{}'.format(epoch, config.num_epochs,
                                                                                             str(total_loss_lst[epoch]),
                                                                                             str(primary_loss_lst[
                                                                                                     epoch]),
                                                                                             str(vgg_loss_lst[epoch])))
                # wandb.log()
            # Save model snapshots every 'snapshot_freq' epochs
            if not os.path.exists(config.snapshots_folder):
                os.mkdir(config.snapshots_folder)

            if epoch % config.snapshot_freq == 0:
                torch.save(self.model, config.snapshots_folder + 'model_epoch_{}.ckpt'.format(epoch))

    @torch.no_grad()
    def eval(self, config, test_dataloader, test_model):
        test_model.eval()
        # Iterate through the test dataloader
        for i, (img, _, name) in enumerate(test_dataloader):
            with torch.no_grad():
                # Move the input image to the appropriate device
                img = img.to(config.device)
                # Generate an output image using the test model
                generate_img = test_model(img)
                # Save the generated image
                torchvision.utils.save_image(generate_img, config.output_images_path + name[0])
        # Calculate SSIM and PSNR measures
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path,
                                                                   config.GTr_test_images_path)
        # Calculate UIQM measures
        UIQM_measures = calculate_UIQM(config.output_images_path)
        return UIQM_measures, SSIM_measures, PSNR_measures


def setup(config):
    # Determine the device based on CUDA availability
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
    # Initialize the model with the specified number of layers and move it to the designated device
    model = UWnet(num_layers=config.num_layers).to(config["device"])
    # Define the transformation for resizing images and converting them to tensors
    transform = transforms.Compose([transforms.Resize((config.resize, config.resize)), transforms.ToTensor()])
    # Create the training dataset using UWNetDataSet with input and label image paths
    train_dataset = UWNetDataSet(config.input_images_path, config.label_images_path, transform, True)
    # Create a dataloader for the training dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False)
    # Print message indicating completion of training dataset reading
    print("Train Dataset Reading Completed.")
    # Initialize the loss function using combinedloss
    loss = combinedloss(config)
    # Initialize the optimizer with Adam and set the learning rate
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    # Initialize the trainer with the model, optimizer, and loss function
    trainer = Trainer(model, opt, loss)
    # If testing is enabled, create the test dataset and dataloader
    if config.test:
        test_dataset = UWNetDataSet(config.test_images_path, None, transform, False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
        print("Test Dataset Reading Completed.")
        # Return training and testing dataloaders along with the model and trainer
        return train_dataloader, test_dataloader, model, trainer
    return train_dataloader, None, model, trainer


def training(config):
    # Initialize Weights & Biases (wandb)
    wandb.init(project="underwater_image_enhancement_UWNet")
    # Update configuration parameters with Weights & Biases (wandb) config
    wandb.config.update(config, allow_val_change=True)
    config = wandb.config
    # Setup training and testing datasets, model, and trainer
    ds_train, ds_test, model, trainer = setup(config)
    # Train the model using the training dataset, configuration, and testing dataset
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
    parser.add_argument('--snapshots_folder', type=str, default="./SnapShot/")
    parser.add_argument('--output_images_path', type=str, default="./ValImg/SRMModel_2/")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)

    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)
    training(config)
