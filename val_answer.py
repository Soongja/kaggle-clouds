import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.rle import rle2mask


fold_df = pd.read_csv('data/folds/5fold_1_classids.csv')
# fold_df = pd.read_csv('data/folds/seg_5fold_3.csv')
fold_df = fold_df.loc[fold_df['split'] == 'val'].reset_index(drop=True)
train_df = pd.read_csv('data/train.csv')


# mask = np.zeros((len(fold_df), 4, 350, 525), dtype=np.uint8)
mask = np.zeros((len(fold_df), 4, 384, 576), dtype=np.uint8)

for idx in tqdm(range(len(fold_df))):
    EncodedPixels = train_df.loc[train_df['Image_Label'].apply(lambda x: x.split('_')[0]) == fold_df['ImageId'][idx]]['EncodedPixels'].values

    if len(EncodedPixels) > 0:
        for i in range(4):
            if str(EncodedPixels[i]) != 'nan':
                mask_c = rle2mask(EncodedPixels[i])
                # mask_c = cv2.resize(mask_c, (525, 350), interpolation=cv2.INTER_NEAREST)
                mask_c = cv2.resize(mask_c, (576, 384), interpolation=cv2.INTER_NEAREST)
                mask[idx,i,:,:] = mask_c

print(mask)
print(np.unique(mask))
# np.save('validation/answer_seg_fold3.npy', mask)
np.save('validation/answer_fold1_384_576.npy', mask)
