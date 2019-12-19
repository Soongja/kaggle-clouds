import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import KFold


train_df = pd.read_csv('train.csv')
train_df['Image'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
print(len(train_df))

image_names = train_df['Image'].values
image_names = np.unique(image_names)
print(len(image_names))

fold_df = pd.DataFrame(columns=['Image', 'split'])
fold_df['Image'] = image_names


apply = 1

os.makedirs('folds', exist_ok=True)

if apply:

    n_splits = 5
    n_fold = 4

    x = fold_df['Image'].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2019)
    kf.get_n_splits(x)

    for fold, (train_index, val_index) in enumerate(kf.split(x)):
        if fold == n_fold:
            print(fold, len(train_index), len(val_index))
            fold_df['split'].iloc[train_index] = 'train'
            fold_df['split'].iloc[val_index] = 'val'

    fold_df.to_csv('folds/%sfold_%s.csv' % (n_splits, n_fold), index=False)

##############################################################