import torch
import os
from PIL import Image


def get_image_list(raw_image_path, clear_image_path, is_train):
    # Initialize an empty list to store image paths.
    image_list = []
    # Retrieve a list of raw image file names from the specified directory.
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    if is_train:
        # For training data, create a list of image paths including both raw and clear images.
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, os.path.join(clear_image_path + image_file), image_file])
    else:
        # For non-training data, create a list of image paths including only raw images.
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, None, image_file])
    return image_list


class UWNetDataSet(torch.utils.data.Dataset):
    """
       Custom dataset class for loading images from specified paths.
       Args:
           raw_image_path (str): Path to the directory containing raw images.
           clear_image_path (str): Path to the directory containing clear images
           transform : Transformations to apply to the images.
           is_train (bool): Flag indicating whether the dataset is for training or not. Defaults to False.
       """
    def __init__(self, raw_image_path, clear_image_path, transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.is_train = is_train
        self.image_list = get_image_list(self.raw_image_path, self.clear_image_path, is_train)
        self.transform = transform

    def __getitem__(self, index):
        """
        Get the item at the specified index.
                Args:
                    index (int): Index of the item to retrieve.
                Returns:
                    tuple: A tuple containing the transformed raw image, transformed clear image, and image name.
        """
        raw_image, clear_image, image_name = self.image_list[index]
        # Open the raw image using PIL's Image.open
        raw_image = Image.open(raw_image)
        # If the dataset is for training:
        if self.is_train:
            # Open the clear image using PIL's Image.open
            clear_image = Image.open(clear_image)
            # Apply transformations to both raw and clear images
            return self.transform(raw_image), self.transform(clear_image), "_"
        return self.transform(raw_image), "_", image_name

    def __len__(self):
        return len(self.image_list)
