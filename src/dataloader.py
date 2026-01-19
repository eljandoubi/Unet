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
        path_to_data,
        train: bool = True,
        image_size: int = 128,
        random_crop_ratio: tuple[float, float] = (0.2, 1),
        inference_mode: bool = False,
    ):
        pass
