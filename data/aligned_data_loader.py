import random
import numpy as np
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.image_folder import *
import sys


class LandmarksDataLoader(BaseDataLoader):
    def __init__(self, opt, data_dir, phase, num_threads, img_a_name, img_b_name, img_c_name):
        dataset = LandmarksFolder(opt=opt, data_dir=data_dir, phase=phase, 
                                img_a_name=img_a_name, img_b_name=img_b_name, img_c_name=img_c_name)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=1, shuffle=(phase=='train'), num_workers=int(num_threads))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'LandmarksDataLoader'
    
    def __len__(self):
        return len(self.dataset)
