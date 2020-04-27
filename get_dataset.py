import os
import pandas as pd
import torch
import torchvision
from PIL import Image

from torch.utils.data import Dataset

from common_constants import NUM_SAMPLE_PER_SCENE, NUM_IMAGE_PER_SAMPLE, IMAGE_NAMES


def convert_map_to_road_map(ego_map):
    mask = (ego_map[0,:,:] == 1) * (ego_map[1,:,:] == 1) * (ego_map[2,:,:] == 1)

    return (~mask)


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


# The dataset class for labeled data.
class LabeledDataset(Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (numpy_array): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, 'scene_{}'.format(scene_id), 'sample_{}'.format(sample_id))

        images = []
        for image_name in IMAGE_NAMES:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)

        return image_tensor, road_image
