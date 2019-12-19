import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.rle import rle2mask


class CloudDataset(Dataset):
    def __init__(self, config, split, transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        self.train_df = pd.read_csv(self.config.TRAIN_DF, engine='python')

        fold_df = pd.read_csv(self.config.FOLD_DF, engine='python')
        self.fold_df = fold_df.loc[fold_df['split'] == self.split].reset_index(drop=True)

        if self.config.DEBUG:
            self.fold_df = self.fold_df[:40]
        print(self.split, 'set:', len(self.fold_df))

        if config.SAMPLER == 'stratified':
            self.labels = self.fold_df['ClassIds'].values

    def __len__(self):
        return len(self.fold_df)

    def __getitem__(self, idx):
        ImageId = self.fold_df["Image"][idx]
        image = cv2.imread(os.path.join(self.config.DATA_DIR, ImageId), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))

        mask = np.zeros((self.config.DATA.IMG_H, self.config.DATA.IMG_W, 4), dtype=np.uint8)
        # mask = np.zeros((1400, 2100, 4), dtype=np.uint8)
        EncodedPixels = self.train_df.loc[self.train_df['Image_Label'].apply(lambda x: x.split('_')[0]) == ImageId]['EncodedPixels'].values

        if len(EncodedPixels) > 0:
            for i in range(4):
                if str(EncodedPixels[i]) != 'nan':
                    mask_c = rle2mask(EncodedPixels[i], shape=(self.config.DATA.IMG_H, self.config.DATA.IMG_W))
                    # mask_c = cv2.resize(mask_c, (self.config.DATA.IMG_W, self.config.DATA.IMG_H), interpolation=cv2.INTER_NEAREST)
                    mask[:,:,i] = mask_c

        # mask의 값은 0과 1!!!!
        mask = mask * 255 # albu 넣을 때 1로 넣어도 되는지 아직 모름

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))
        # mask = cv2.resize(mask, (self.config.DATA.IMG_W, self.config.DATA.IMG_H), interpolation=cv2.INTER_NEAREST)

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            # transforms.Normalize(mean=[0.261, 0.279, 0.327],
            #                      std=[0.254, 0.249, 0.235]),
        ])
        image = normalize(image)

        mask = mask / 255.

        mask = torch.from_numpy(mask).permute((2, 0, 1)).float()

        return image, mask
