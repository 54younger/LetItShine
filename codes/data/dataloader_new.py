import numpy as np
import pandas as pd 
from PIL import Image
import os
import tqdm
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from .view_generator import ContrastiveLearningViewGenerator
from .gaussian_blur import GaussianBlur
from torchvision.transforms import functional as F
import torch.multiprocessing
import warnings
torch.multiprocessing.set_sharing_strategy('file_system')

class Misaligned_images:
    def __init__(self, shift_level=2,size=224):
        self.shift_level = shift_level
        self.size = size

    def shift_ij(self,image_height, image_width, i, j, h, w):
        k=self.shift_level
        direction = random.choice([0, 1])
        if direction == 0:
            new_i = random.choice([i + k, i - k])

            if new_i < 0:
                new_i = 0
            elif new_i > image_height - h:
                new_i = image_height - h

            new_j = j

        else:
            new_j = random.choice([j + k, j - k])
            
            # 确保新的j不超出边界
            if new_j < 0:
                new_j = 0
            elif new_j > image_width - w:
                new_j = image_width - w

            new_i = i
        # print(f"New i: {new_i}, New j: {new_j}")
        return new_i,new_j,h,w

        
    def __call__(self, image):
        if self.shift_level<0 or self.shift_level>16:
            print("0<=shift_level<=16")
            return None
        img1 = image[:3, :, :] #BF
        img2 = image[3:, :, :] #FL
    
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(self.size,self.size))
        #print(f"Crop coordinates: top={i}, left={j}, height={h}, width={w}")
        img_cropped1 = transforms.functional.crop(img1,i,j,h,w)
        i, j, h, w = self.shift_ij(256, 256, i, j, h, w)
        #print(f"top={i}, left={j}, height={h}, width={w}")
        img_cropped2 = transforms.functional.crop(img2,i,j,h,w)
        combined_tensor = torch.cat((img_cropped1, img_cropped2), dim=0)
        return combined_tensor


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
        transform_list = [v2.RandomPosterize(3), v2.GaussianBlur(5, 1.5), v2.RandomSolarize(100)]
        self.normalize_BF = v2.Normalize((0.5251, 0.5998, 0.6397), (0.2339, 0.1905, 0.1573))
        #self.normalize_FL = v2.Normalize((0.0804, 0.0489, 0.1264, 0.1098), (0.0732, 0.0579, 0.0822, 0.0811))
        self.normalize_FL = v2.Normalize(
    mean=(0.0804, 0.0489, 0.1264),
    std=(0.0732, 0.0579, 0.0822)
)

#############
        if split == 'train':
            self.transform_train = v2.Compose([
                v2.RandomHorizontalFlip(),
               # v2.RandomResizedCrop(size=size, scale=(1.0, 1.0), antialias=True),
               # v2.CenterCrop(size=size),
                #v2.RandomCrop(size=size),
                Misaligned_images(16,size),
            ])
            self.transform_BF = v2.Compose([
                # v2.GaussianBlur(5, 1.5),
                v2.RandomChoice(transform_list, p=[0.4, 0.2, 0.4]),
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
            self.transform_train = v2.Compose([
                Misaligned_images(16,size),
            ])
            self.transform_BF = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.normalize_BF,
                #v2.Resize(size=size, antialias=True),
                #v2.CenterCrop(size=size),
                #v2.RandomCrop(size=size),
            ])
            self.transform_FL = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.normalize_FL,
                #v2.Resize(size=size, antialias=True),
                #v2.CenterCrop(size=size),
                #v2.RandomCrop(size=size),
            ])

    @staticmethod
    def get_simclr_pipeline_transform(size, normalize_fn=v2.Normalize((0.5,), (0.5,)), c=3, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        print("using new data augmentation...")
        if c==4:
            color_jitter = MultiChannelColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.5 * s)
            data_transforms = transforms.Compose([
                                              v2.RandomApply([color_jitter], p=0.95),
                                              # GaussianBlur(kernel_size=int(0.05 * size), c=c),
                                              v2.ToImage(), v2.ToDtype(torch.float32, scale=True), 
                                              v2.GaussianBlur(5, sigma=(0.3, 3.2)), normalize_fn,
                                              v2.RandomHorizontalFlip(),
                                              v2.RandomResizedCrop(size=size,scale=(1.0, 1.0), antialias=True),
                                              ])
        else:
            color_jitter = transforms.ColorJitter(0.5 * s, 0.2 * s, 0.2 * s, 0.2 * s)
            data_transforms = transforms.Compose([
                                            v2.RandomApply([color_jitter], p=0.95),
                                            # GaussianBlur(kernel_size=int(0.05 * size), c=c),
                                            v2.GaussianBlur(5, sigma=(0.3, 2.0)),
                                            v2.ToImage(), v2.ToDtype(torch.float32, scale=True), normalize_fn, 
                                            v2.RandomHorizontalFlip(),
                                            v2.RandomResizedCrop(size=size,scale=(1.0, 1.0), antialias=True),
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
            #slide = data['Slide']
            #patch = data['Patch']
            name = data['Name']
            BF_image, FL_image, MM_image = None, None, None
            if self.mode in ['BF', 'MM']:
                BF_path = os.path.join(self.root, f'/mnt/d/cache/BF/{name}.pt')
                BF_image = torch.load(BF_path)
                if self.mode == 'BF':
                    FL_image, MM_image = BF_image, BF_image

            if self.mode in ['FL', 'MM']:
                FL_path = os.path.join(self.root, f'/mnt/d/cache/FL/{name}.pt')
                FL_image = torch.load(FL_path)
                if self.mode == 'FL':
                    BF_image, MM_image = FL_image, FL_image


            if self.mode == 'MM':
                MM_image = torch.cat([BF_image, FL_image], dim=0)


            if self.split == 'train':
                BF_image = self.transform_train(BF_image)
                FL_image = self.transform_train(FL_image)
                MM_image = self.transform_train(MM_image)
            if self.split == 'test':
                MM_image = self.transform_train(MM_image)

            label = data['Diagnosis']
            name = data['Name']
            data_dict = {'BF': BF_image, 'FL': FL_image, 'MM': MM_image, 'label': label, 'name': name}
            if self.split == 'cl':
                return data_dict[self.mode], data_dict['label']
            else:
                return data_dict
        else:
            print("Dataloader error: Dataframe (df) can't be None!!!")



