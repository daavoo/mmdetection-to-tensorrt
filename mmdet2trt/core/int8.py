import os
import torch.utils.data as data

from mmdet.apis.inference import LoadImage
from mmdet.pipelines import Compose


class Int8CalibrationDataset(data.Dataset):
    
    def __init__(self, folder, pipeline):
        self.images = os.listdir(folder)
        self.pipeline = Compose([LoadImage()] + cfg.data.test.pipeline[1:])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.pipeline(self.images[idx])
