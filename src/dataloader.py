import os
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class ADE20KDataset(Dataset):
    def __init__(
        self,
        path_to_data: str,
        train: bool = True,
        image_size: int = 128,
        random_crop_ratio: tuple[float, float] = (0.2, 1),
        inference_mode: bool = False,
    ):
        self.path_to_data = path_to_data
        self.inference_mode = inference_mode
        self.train = train
        self.image_size = image_size
        self.min_ratio, self.max_ratio = random_crop_ratio

        if train:
            split = "training"
        else:
            split = "validation"

        ### Get Path to Images and Segmentations ###
        self.path_to_images = os.path.join(self.path_to_data, "images", split)
        self.path_to_annotations = os.path.join(self.path_to_data, "annotations", split)

        ### Get All Unique File Roots ###
        self.file_roots = [
            path.split(".")[0] for path in os.listdir(self.path_to_images)
        ]

        ### Store all Transforms we want ###
        self.resize = transforms.Resize(size=(self.image_size, self.image_size))
        self.normalize = transforms.Normalize(
            mean=(0.48897059, 0.46548275, 0.4294),
            std=(0.22861765, 0.22948039, 0.24054667),
        )
        self.random_resize = transforms.RandomResizedCrop(
            size=(self.image_size, self.image_size)
        )
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_roots)
