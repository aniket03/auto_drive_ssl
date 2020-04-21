import os
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset

from common_constants import NUM_SAMPLE_PER_SCENE, NUM_IMAGE_PER_SAMPLE, IMAGE_NAMES


# The dataset class for unlabeled data.
class UnlabeledDataset(Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (numpy array): a list of scene indices for the unlabeled data
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform

        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, 'scene_{}'.format(scene_id),
                                       'sample_{}'.format(sample_id))

            images = []
            for image_name in IMAGE_NAMES:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                images.append(self.transform(image))
            image_tensor = torch.stack(images)

            return image_tensor, scene_id, index

        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = IMAGE_NAMES[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, 'scene_{}'.format(scene_id),
                                      'sample_{}'.format(sample_id), image_name)

            image = Image.open(image_path)

            return self.transform(image), index % NUM_IMAGE_PER_SAMPLE

