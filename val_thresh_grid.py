import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def dice_coef_numpy(preds, labels):
    smooth = 1e-6
    intersection = np.sum(np.float32(preds) * np.float32(labels), axis=(1,2))
    union = np.sum(np.float32(preds), axis=(1,2)) + np.sum(np.float32(labels), axis=(1,2))
    # class 별로 찍게 하자 [N, C]
    dice = np.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    return dice


def draw_convex_hull(mask, mode='convex'):
    # img = np.zeros(mask.shape)
    img = np.zeros_like(mask)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode == 'rect':  # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), 1, -1)
        elif mode == 'convex':  # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, 1, -1)
        elif mode == 'approx':
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(img, [approx], 0, 1, -1)
        else:  # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, 1, -1)
    return img


def post_process(probability, test_image, threshold, min_size, min_coverage, fill_up=False, convex=False, black_test_image=False):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    mask = np.uint8(mask)
    predictions = cv2.resize(mask, (525, 350), interpolation=cv2.INTER_NEAREST)

    if fill_up:
        contours, _ = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_up_predictions = np.zeros((350, 525), np.uint8)
        for c in contours:
            cv2.drawContours(filled_up_predictions, [c], 0, 1, -1)
        predictions = filled_up_predictions

    if convex:
        predictions = draw_convex_hull(predictions, mode='convex')

    num_component, component = cv2.connectedComponents(predictions.astype(np.uint8))
    predictions = np.zeros((350, 525), np.uint8)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1

    if black_test_image:
        predictions[test_image <= 2] = 0

    if np.sum(predictions) / (350*525) < min_coverage:
        predictions[:,:] = 0

    return predictions

################################

out_dir = 'grid_search/fold1_ensemble4'
os.makedirs(out_dir, exist_ok=True)

# pred = np.load('validation/fold1_ensemble2_0.6633_0.6648.npy')
# pred = np.load('validation/fold1_ensemble3_0.6633_0.6648_0.6624.npy')
pred = np.load('validation/fold1_ensemble4_0.6633_0.6648_0.6624_0.6624.npy')

answer = np.load('validation/answer_fold1.npy')

fold_df_path = 'data/folds/5fold_1_classids.csv'

################################

# threshold = [0.5,0.5,0.5,0.5]

fill_up = False
convex = False

# min_size = [4000,4000,4000,4000]

black_test_image = True

# min_coverage = [0.05,0.05,0.05,0.05]

################################

fold_df = pd.read_csv(fold_df_path, engine='python')
fold_df = fold_df.loc[fold_df['split'] == 'val'].reset_index(drop=True)
img_names = fold_df['ImageId'].values

for i, cls in enumerate(['Fish', 'Flower', 'Gravel', 'Sugar']):

    df = pd.DataFrame(columns=['threshold', 'min_size', 'min_coverage', 'dice'])
    # df = pd.read_csv(os.path.join(out_dir, '%s.csv' % cls))

    # thresholds = [0.5]
    # min_sizes = [4000]
    # min_coverages = [0.05]

    thresholds = [0.3,0.325,0.35,0.375,0.4,0.425,0.45]  # 7
    min_sizes = [4000]  # 1
    min_coverages = [0.04,0.06,0.08,0.1]  # 28

    ################################

    count = 0
    # count = len(df)
    for threshold in thresholds:
        for min_size in min_sizes:
            for min_coverage in min_coverages:

                pred_processed = np.zeros((pred.shape[0], 350, 525), np.uint8)

                for idx in tqdm(range(answer.shape[0])):

                    val_image = cv2.imread(os.path.join('data/train_images', img_names[idx]), 0)
                    val_image = cv2.resize(val_image, (525, 350))
                    # val_image = np.float32(val_image)

                    pred_processed[idx] = post_process(pred[idx,i], val_image, threshold=threshold,
                                                             min_size=min_size, min_coverage=min_coverage,
                                                             fill_up=fill_up, convex=convex, black_test_image=black_test_image)
                    # cv2.imshow('answer', answer[idx,i] * 255)
                    # cv2.imshow('pred', pred_processed[idx,i] * 255)
                    # cv2.waitKey()

                dice = dice_coef_numpy(pred_processed, answer[:,i])
                print(count, threshold, min_size, min_coverage, dice)
                # dice_total = np.mean(dice)
                # print(dice_total)

                df.loc[count] = [threshold, min_size, min_coverage, dice]
                count += 1

    df = df.astype({"min_size": int})
    df = df.sort_values(["dice"], ascending=[False])
    df.to_csv(os.path.join(out_dir, '%s.csv' % cls), index=False)
