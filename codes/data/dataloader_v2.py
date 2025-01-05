import numpy as np
import pandas as pd 
from PIL import Image
import os
import tqdm
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from .view_generator import ContrastiveLearningViewGenerator
from .gaussian_blur import GaussianBlur
from torchvision.transforms import functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class MultiChannelGreyScale:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, image):
        if torch.rand(1).item() < self.p:
            enhanced_channels = [F.rgb_to_grayscale(c) for c in image.split()]
        else:
            return image
        return Image.merge(image.mode, tuple(enhanced_channels))

class MultiChannelColorJitter:
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.1, hue=0.5):
        self.color_transform = v2.ColorJitter(brightness=brightness, 
                                              contrast=contrast, 
                                              saturation=saturation,
                                              hue=hue)
        
    def __call__(self, image):
        enhanced_channels = [self.color_transform(c) for c in image.split()]
        return Image.merge(image.mode, enhanced_channels)

class MultiModalCancerDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, root_path, df, mode='BF', split='train', size=256, n_views=2):
        # mode: BF, FL, MM (multi-modal) 
        # Passing the path to the train csv file reads the data from the csv with the labels
        # If None is passes insted only the images in the image folder is loaded (wich is useful for the test set)
        
        self.root = root_path
        self.df = df
        self.mode = mode
        self.split = split
        self.normalize_BF = v2.Normalize((0.5251, 0.5998, 0.6397), (0.2339, 0.1905, 0.1573))
        self.normalize_FL = v2.Normalize((0.0804, 0.0489, 0.1264, 0.1098), (0.0732, 0.0579, 0.0822, 0.0811))
#############
        if split == 'train':
            self.transform_train = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomResizedCrop(size=size, scale=(0.5, 1.0), antialias=True),
            ])
            self.transform_BF = transforms.Compose([
                v2.GaussianBlur(5, 1.5),
                v2.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.normalize_BF,
            ])
            self.transform_FL = v2.Compose([
                MultiChannelColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.GaussianBlur(5, sigma=(0.3, 3.2)),
                self.normalize_FL,
            ])
        elif split == 'cl': # contrastive learning
            self.transform_BF = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(size, self.normalize_BF, c=3), n_views)
            self.transform_FL = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(size, self.normalize_FL, c=4), n_views)
        else:
            self.transform_BF = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.normalize_BF,
                v2.Resize(size=size, antialias=True),
            ])
            self.transform_FL = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.normalize_FL,
                v2.Resize(size=size, antialias=True),
            ])

    @staticmethod
    def get_simclr_pipeline_transform(size, normalize_fn=v2.Normalize((0.5,), (0.5,)), c=3, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        print("using new data augmentation...")
        if c==4:
            color_jitter = MultiChannelColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.5 * s)
            data_transforms = transforms.Compose([
                                              v2.RandomApply([color_jitter], p=0.9),
                                              GaussianBlur(kernel_size=int(0.05 * size), c=c),
                                              v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize_fn,
                                              v2.RandomHorizontalFlip(),
                                              v2.RandomResizedCrop(size=size,scale=(0.5, 1.0), antialias=True),
                                              ])
        else:
            color_jitter = transforms.ColorJitter(0.5 * s, 0.2 * s, 0.2 * s, 0.2 * s)
            data_transforms = transforms.Compose([
                                            v2.RandomApply([color_jitter], p=0.9),
                                            GaussianBlur(kernel_size=int(0.05 * size), c=c),
                                            v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize_fn, 
                                            v2.RandomHorizontalFlip(),
                                            v2.RandomResizedCrop(size=size,scale=(0.5, 1.0), antialias=True),
                                            ])
        return data_transforms
    
    def __len__(self):
        return len(self.df)

    def get_ratio(self):
        ratio_diagnosis = torch.tensor(self.df['diagnosis'].value_counts(normalize=True))
        return ratio_diagnosis[1],ratio_diagnosis[0] 

    def __getitem__(self, idx):
        
        if self.root is not None and self.df is not None:
            data = self.df.iloc[idx]
            BF_image, FL_image, MM_image = None, None, None
            if self.mode in ['BF', 'MM']:
                BF_path = os.path.join(self.root, data['BF_path'])
                BF_image = Image.open(BF_path)
                BF_image = self.transform_BF(BF_image)
                if self.mode == 'BF':
                    FL_image, MM_image = BF_image, BF_image

            if self.mode in ['FL', 'MM']:
                FL_path = os.path.join(self.root, data['FL_path'])
                FL_image = Image.open(FL_path)
                # FL_image = FL_image.convert("RGB")
                FL_image = self.transform_FL(FL_image)
                if self.mode == 'FL':
                    BF_image, MM_image = FL_image, FL_image

            if self.mode == 'MM':
                if self.split == 'cl':
                    MM_image = [torch.cat([bf, fl], dim=0) for bf, fl in zip(BF_image, FL_image)]
                else:
                    MM_image = torch.cat([BF_image, FL_image], dim=0)

            if self.split == 'train':
                BF_image = self.transform_train(BF_image)
                FL_image = self.transform_train(FL_image)
                MM_image = self.transform_train(MM_image)

            label = data['Diagnosis']
            name = data['Names']
            data_dict = {'BF': BF_image, 'FL': FL_image, 'MM': MM_image, 'label': label, 'name': name}
            if self.split == 'cl':
                return data_dict[self.mode], data_dict['label']
            else:
                return data_dict
        else:
            print("Dataloader error: Dataframe (df) can't be None!!!")


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=16, length=24):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        if np.random.uniform() > 0.5:
            return img

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            x_length = np.random.randint(4, self.length)
            y_length = np.random.randint(4, self.length)

            y1 = np.clip(y - y_length // 2, 0, h)
            y2 = np.clip(y + y_length // 2, 0, h)
            x1 = np.clip(x - x_length // 2, 0, w)
            x2 = np.clip(x + x_length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Noise(object):
    def __init__(self, level=30):
        self.level = level / 255.

    def __call__(self, img):
        if np.random.uniform() > 0.9:
            return img

        noise = torch.randn_like(img)
        img = img + noise * self.level
        return img

class MultiNoises(object):
    def __init__(self,noise_type = 'guass'):
        self.noise_type = noise_type

    def __call__(self, image):
        if self.noise_type == 'gauss':
            return self.add_gaussian_noise(image)
        elif self.noise_type == 's&p':
            return self.add_salt_and_pepper_noise(image)
        elif self.noise_type == 'poisson':
            return self.add_poisson_noise(image)
        elif self.noise_type == 'speckle':
            return self.add_speckle_noise(image)
        else:
            raise ValueError("Unsupported noise type")

    def add_gaussian_noise(self,image, mean=0, var=0.1):
        row, col = image.size
        ch =4
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return Image.fromarray(noisy)

    def add_salt_and_pepper_noise(self,image, s_vs_p=0.5, amount=0.004):
        row, col = image.size
        ch = 4
        out = np.copy(image)
        size = row * col * ch
        # Salt mode
        num_salt = np.ceil(amount * float(size) * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in (col, row)]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * float(size) * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in (col, row)]
        out[coords] = 0

        return Image.fromarray(out)

    def add_poisson_noise(self,image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return Image.fromarray(noisy)

    def add_speckle_noise(self,image):
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return Image.fromarray(noisy)
