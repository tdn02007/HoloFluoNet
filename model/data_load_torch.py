import os

from glob import glob
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class DatasetLoader(Dataset):
    def __init__(self, is_train):

        self.is_train = is_train

        if self.is_train:
            self.input_dir1 = "../data/train_data/phase/"
            self.label_dir1 = "../data/train_data/background/"
            self.label_dir2 = "../data/train_data/live/"
            self.label_dir3 = "../data/train_data/dead/"
            self.label_dir4 = "../data/train_data/nuclei/"
            self.distance_dir1 = "../data/train_data/distance/"
        else:
            self.input_dir1 = "../data/test_data/phase/"
            self.label_dir1 = "../data/test_data/background/"
            self.label_dir2 = "../data/test_data/live/"
            self.label_dir3 = "../data/test_data/dead/"
            self.label_dir4 = "../data/test_data/nuclei/"
            self.distance_dir1 = "../data/test_data/distance/"

        self.input_images1 = sorted(os.listdir(self.input_dir1))
        self.label_images1 = sorted(os.listdir(self.label_dir1))
        self.label_images2 = sorted(os.listdir(self.label_dir2))
        self.label_images3 = sorted(os.listdir(self.label_dir3))

        self.distance_images1 = sorted(os.listdir(self.distance_dir1))

    @classmethod
    def preprocess(cls, pil_img, train):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if train:
            img_trans = img_trans / 255.

        return img_trans
    
    @classmethod
    def one_hot_encoder(cls, np_img):

        max_positions = np.argmax(np_img, axis=0)
        one_hot_image = np.zeros_like(np_img)
        rows, cols = np.indices(max_positions.shape)
        one_hot_image[max_positions, rows, cols] = 1

        return one_hot_image

    def __len__(self):
        return len(self.input_images1)

    def __getitem__(self, index):
        image_name1 = self.input_images1[index]

        input_file1 = glob(self.input_dir1 + image_name1)
        input_image = np.array(Image.open(input_file1[0]).resize((256, 256)))
        input_image = self.preprocess(input_image, True)

        label_name1 = self.label_images1[index]
        label_name2 = self.label_images2[index]
        label_name3 = self.label_images3[index]
        label_name4 = self.label_images3[index]

        label_file1 = glob(self.label_dir1 + label_name1)
        label_file2 = glob(self.label_dir2 + label_name2)
        label_file3 = glob(self.label_dir3 + label_name3)
        label_file4 = glob(self.label_dir4 + label_name4)

        label_image1 = np.array(Image.open(label_file1[0]).resize((256, 256)))
        label_image2 = np.array(Image.open(label_file2[0]).resize((256, 256)))
        label_image3 = np.array(Image.open(label_file3[0]).resize((256, 256)))
        label_image4 = np.array(Image.open(label_file4[0]).resize((256, 256)))

        label_image = np.stack([label_image1, label_image2, label_image3], axis=0)

        label_image = self.one_hot_encoder(label_image)

        label_image4 = np.expand_dims(np.where(label_image4 == 255, 1, 0).astype(np.float32), axis=0)

        label_image = np.concatenate((label_image, label_image4), axis=0)

        distance_name1 = self.distance_images1[index]
        distance_file1 = glob(self.distance_dir1 + distance_name1)
        distance_image1 = np.array(Image.open(distance_file1[0]).resize((256, 256)))

        distance_image1 = self.preprocess(distance_image1, True)

        return {
            "input_image": torch.from_numpy(input_image).float(),
            "label_image": torch.from_numpy(label_image).float(),
            "distance_image": torch.from_numpy(distance_image1).float(),
        }