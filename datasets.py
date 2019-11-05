import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import torchvision
import skimage.io as io
import skimage.transform as transform
import numpy as np


class CaptionDataset(Dataset):
    """A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches."""

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.tensor(self.imgs[i // self.cpi] / 255., dtype=torch.float)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.tensor(self.captions[i], dtype=torch.long)

        caplen = torch.tensor([self.caplens[i]], dtype=torch.long)

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.tensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)], dtype=torch.long)
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class TestFolder(Dataset):
    """Image folder for testing of the model."""

    def __init__(self, dir: str, transform: bool = None, size: (int, tuple, list) = 256):
        super().__init__()
        self.dir = dir

        if isinstance(size, int):
            size = size, size
        else:
            assert isinstance(size, (tuple, list)) and len(size) == 2
        self.size = size

        self.names = []
        if os.path.isdir(dir):
            names = os.listdir(dir)

            for name in names:
                if name.split('.')[-1] in {'jpeg', 'jpg', 'png', 'tif'}:
                    self.names.append(name)

            assert self.names
        else:
            self.names = [self.dir]
            self.dir = ''

        self.transform = transform

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.names[idx])
        img = io.imread(path)
        if self.size:
            img = transform.resize(img, self.size)  # Convrt to (0, 1) automatically
        else:
            img = img.astype(np.float) / 255

        if len(img.shape) == 2:  # For gray scale color img
            img = img[..., np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        else:
            img = img.transpose((2, 0, 1))  # convert to torch style channel first img
        img = torch.tensor(img, dtype=torch.float)

        if self.transform:
            img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.names)
