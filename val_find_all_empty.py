import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dice_coef_numpy(preds, labels):
    smooth = 1e-6
    intersection = np.sum(np.float32(preds) * np.float32(labels), axis=(2, 3))
    union = np.sum(np.float32(preds), axis=(2,3)) + np.sum(np.float32(labels), axis=(2,3))
    # class 별로 찍게 하자 [N, C]
    dice = np.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    return dice


################################

fold_df = pd.read_csv('data/folds/5fold_1.csv', engine='python')
fold_df = fold_df.loc[fold_df['split'] == 'val'].reset_index(drop=True)
img_names = fold_df['Image'].values

answer = np.load('validation/val_5fold_1.npy')
pred = np.load('validation/fold1_ensemble2_0.6633_0.6648.npy')
pred_processed = np.load('validation/fold1_post_processed/base.npy') # [1109,4,350,525]
pred_04 = np.load('validation/fold1_post_processed/0.4.npy') # [1109,4,350,525]

################################

all_empty_count = 0
all_empty_idxs = []
for i in range(pred_04.shape[0]):
    if np.sum(pred_04[i]) == 0:
        all_empty_count += 1
        all_empty_idxs.append(i)

print(all_empty_count)
print(all_empty_idxs)

################################

answer_all_empty = answer[all_empty_idxs]
print(answer_all_empty.shape, answer_all_empty.dtype)

pred_all_empty = pred[all_empty_idxs]
print(pred_all_empty.shape, pred_all_empty.dtype)

################################

'''
values = []
for i in range(4):
    class_values = []
    for j in tqdm(range(answer_all_empty.shape[0])):
        _answer = answer_all_empty[j,i]
        _pred = pred_all_empty[j,i]
        _pred = cv2.resize(_pred, (525,350), interpolation=cv2.INTER_LINEAR)

        _pred = cv2.bitwise_and(_pred,_pred, mask=_answer)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x_data, y_data = np.meshgrid(np.arange(_pred.shape[1]),
        #                              np.arange(_pred.shape[0]))
        # x_data = x_data.flatten()
        # y_data = y_data.flatten()
        # z_data = _pred.flatten()
        # ax.bar3d(x_data,
        #          y_data,
        #          np.zeros(len(z_data)),
        #          1, 1, z_data)
        # plt.show()


        _pred = _pred.reshape(-1)

        for k in range(350*525):
            if _pred[k] != 0:
                class_values.append(_pred[k])
    # print(class_values)

    plt.hist(class_values, bins=100, range=(0,0.5))
    plt.show()
    # values.append(class_values)

# print(values)
'''

dice = dice_coef_numpy(pred_processed, answer)
dice_total = np.mean(dice)
print(dice_total)

pred_processed[all_empty_idxs] = pred_04[all_empty_idxs]
dice = dice_coef_numpy(pred_processed, answer)
dice_total = np.mean(dice)
print(dice_total)
