from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import re


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
                            # (split files are called 'train.txt' and 'test.txt')
        self.__root = root

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        self.__data = list()
        self.labels = dict()

        i = 0
        with open(root + '/' + split + '.txt', 'r') as f:
            line = f.readline()
            label = re.split('/', line)[0]

            if line.find('BACKGROUND_Google') == -1:
                if label not in self.labels.keys():
                    self.labels[label] = i
                    i += 1
                self.__data.append(line)

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int
        image = pil_loader(self.__root + '101_ObjectCategories/' + self.__data[index])
        label = re.split('/', self.__data[index])[0]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        ret = (image, self.labels[label])
        return ret

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.__data)  # Provide a way to get the length (number of elements) of the dataset
        return length
