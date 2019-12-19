import os
import shutil
import numpy as np
import math
import cv2
import multiprocessing as mp
import pandas as pd
from utils.rle import rle2mask,mask2rle


ORI_SIZE = (1400, 2100) # (height, width)
NEW_SIZE = (384, 576) # (height, width)


train_base_path = 'data/train_images/'
test_base_path = 'data/test_images/'
train_images_dest_path = 'data/train_images_384_576/'
test_images_dest_path = 'data/test_images_384_576/'


# Data pre-process
def preprocess_image(image_id, base_path, save_path, HEIGHT, WIDTH):
    image = cv2.imread(base_path + image_id)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    cv2.imwrite(save_path + image_id, image)


def pre_process_set(df, preprocess_fn):
    n_cpu = mp.cpu_count()
    df_n_cnt = df.shape[0] // n_cpu
    pool = mp.Pool(n_cpu)

    dfs = [df.iloc[df_n_cnt * i:df_n_cnt * (i + 1)] for i in range(n_cpu)]
    dfs[-1] = df.iloc[df_n_cnt * (n_cpu - 1):]
    res = pool.map(preprocess_fn, [x_df for x_df in dfs])
    pool.close()


def pre_process_rle_update():
    df = pd.read_csv(os.path.join('data', 'train.csv'))
    for idx, row in df.iterrows():
        ######## blocked ##########
        # this is for erase black part (labeeling noise)
        # image_name = row[0].split("_")[0]
        # train_image = cv2.imread(os.path.join(train_base_path,image_name),-1)
        # train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
        # train_image = cv2.resize(train_image, (576, 384), interpolation=cv2.INTER_LINEAR)
        ######## blocked ##########

        encodedpixels = row[1]
        if encodedpixels is not np.nan:
            mask = rle2mask(encodedpixels)
            mask = cv2.resize(mask, (NEW_SIZE[1], NEW_SIZE[0]), interpolation=cv2.INTER_NEAREST)
            # mask[train_image <= 2] = 0
            rle = mask2rle(mask)
            df.at[idx, 'EncodedPixels'] = rle
    df.to_csv('data/train_384_576.csv', index=False)

def preprocess_data(df, HEIGHT=NEW_SIZE[0], WIDTH=NEW_SIZE[1]):
    '''
    This function needs to be defined here, because it will be called with no arguments,
    and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
    '''
    df = df.reset_index()
    for i in range(df.shape[0]):
        item = df.iloc[i]
        image_id = item['ImageId']
        item_set = item['split']
        if item_set in['train','val']:
            preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
        else: ## test
            preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)


if __name__ == "__main__":
    # Pre-procecss train/test set
    apply = 1
    if apply:
        print("train_prepare")
        X_train = os.listdir(train_base_path)
        split = ['train' for i in range(len(X_train))]
        X_train = pd.DataFrame(list(zip(X_train, split)), columns=['ImageId', 'split'])
        if os.path.exists(train_images_dest_path):
            shutil.rmtree(train_images_dest_path)

        # Creating train, validation and test directories
        os.makedirs(train_images_dest_path, exist_ok=True)
        pre_process_set(X_train, preprocess_data)

    apply = 1
    if apply:
        print("test_prepare")
        X_test = os.listdir(test_base_path)
        split = ['test' for i in range(len(X_test))]
        X_test = pd.DataFrame(list(zip(X_test,split)),columns=['ImageId','split'])
        if os.path.exists(test_images_dest_path):
            shutil.rmtree(test_images_dest_path)

        os.makedirs(test_images_dest_path, exist_ok=True)
        pre_process_set(X_test, preprocess_data)

    apply = 0
    if apply:
        print("rle_mask_prepare")
        # Update rle mask.
        pre_process_rle_update()

    print("success")