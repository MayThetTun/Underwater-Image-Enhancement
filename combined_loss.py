from VGG_loss import *
from torchvision import models


class combinedloss(nn.Module):
    def __init__(self, config):
        super(combinedloss, self).__init__()
        # Load pre-trained VGG19 model with batch normalization
        vgg = models.vgg19_bn(pretrained=True)
        print("VGG model is loaded")
        # Initialize VGG loss with the loaded VGG model and provided configuration
        self.vggloss = VGG_loss(vgg, config)
        # Freeze parameters of the VGG loss to prevent training
        for param in self.vggloss.parameters():
            param.requires_grad = False
        # Initialize Mean Squared Error (MSE) loss and L1 loss, moving them to the specified device
        self.mseloss = nn.MSELoss().to(config['device'])
        self.l1loss = nn.L1Loss().to(config['device'])

    def forward(self, out, label):
        # Compute VGG loss for the generated output and the ground truth label
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        # Calculate mean squared error loss
        mse_loss = self.mseloss(out, label)
        # Compute L1 loss between VGG features of output and label
        vgg_loss = self.l1loss(inp_vgg, label_vgg)
        # Total loss is the sum of mean squared error loss and VGG loss
        total_loss = mse_loss + vgg_loss
        return total_loss, mse_loss, vgg_loss
