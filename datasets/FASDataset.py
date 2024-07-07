import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FASDataset(Dataset):
    """ A data loader for Face PAD where samples are organized in this way
    Args:
        test_list (string): The test data list
        map_size (int): size of pixel-wise binary supervision map. The paper uses map_size=14
        transform: A function/transform that takes in a sample and returns a transformed version
        smoothing (bool): Use label smoothing
    """

    def __init__(self, test_list, depth_map_size, transform, smoothing):
        super().__init__()
        self.test_list = test_list
        with open(test_list, 'r') as f:
            self.data = f.readlines()
        self.depth_map_size = depth_map_size
        self.transform = transform

        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99

    def __getitem__(self, index):
        """ Get image, output map and label for a given index
        Args:
            index (int): index of image
        Returns:
            img (PIL Image):
            mask: output map (32x32)
            label: 1 (genuine), 0 (fake)
        """
        img_name, label = self.data[index].strip('\n').split('\t')

        img = Image.open(img_name)

        label = np.array(int(label)).astype(np.float32)
        label = np.expand_dims(label, axis=0)

        if label == 1:
            depth_map = np.ones((self.depth_map_size[0], self.depth_map_size[1]), 
                                dtype=np.float32) * self.label_weight
        else:
            depth_map = np.ones((self.depth_map_size[0], self.depth_map_size[1]), 
                                dtype=np.float32) * (1.0 - self.label_weight)

        if self.transform:
            img_transform = self.transform(img)

        img = np.array(transforms.Resize((128, 128))(img))

        return img_transform, depth_map, label, img_name, img

    def __len__(self):
        return len(self.data)
