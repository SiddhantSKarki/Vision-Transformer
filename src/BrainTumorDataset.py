from torch.utils import data
from sklearn.model_selection import train_test_split

import os
import pandas as pd
from PIL import Image



class BrainTumorDataset(data.Dataset):
    def __init__(self, data_dir, train=True, test_size=0.2, transform=None, random_state=42):
        labels = os.listdir(data_dir)
        self.transform = transform
        self.data_dir = data_dir
        self.class_encoding = {k:v for k,v in enumerate(labels)}
        
        images = []
        for i in range(len(labels)):
            cur_dir_imgs = os.listdir(os.path.join(data_dir, labels[i]))
            cur_dir_imgs = [os.path.join(data_dir,labels[i], curr) for curr in cur_dir_imgs]
            images += list(zip(cur_dir_imgs, [i]*len(cur_dir_imgs)))
            
        data = pd.DataFrame(images, columns=["image", "class"])
        tr_x, ts_x, tr_y, ts_y = train_test_split(data[["image"]], data[['class']],
                                                 test_size=test_size, stratify=data[["class"]],
                                                 random_state=random_state)
        self.train = pd.concat([tr_x, tr_y], axis=1)
        self.test = pd.concat([ts_x, ts_y], axis=1)
        self.indexer = self.train if train else self.test
        
    def __len__(self):
        return self.indexer.shape[0]
        
    def __getitem__(self, idx):
        image_dir, label = self.indexer.iloc[idx]
        image = Image.open(image_dir)
        if self.transform is not None:
            image = self.transform(image)
        return image, label