import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def draw_convex_hull(mask, mode='convex'):
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode == 'rect':  # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        elif mode == 'convex':  # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255), -1)
        elif mode == 'approx':
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255), -1)
        else:  # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
    return img / 255.


def post_process(probability, test_image, threshold, min_size, min_coverage, fill_up=False):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    mask = np.uint8(mask)
    predictions = cv2.resize(mask, (525, 350), interpolation=cv2.INTER_NEAREST)

    # experiments
    if fill_up:
        contours, _ = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_up_predictions = np.zeros((350, 525), np.uint8)
        for c in contours:
            cv2.drawContours(filled_up_predictions, [c], 0, 1, -1)
        predictions = filled_up_predictions

    num_component, component = cv2.connectedComponents(predictions.astype(np.uint8))
    _predictions = np.zeros((350, 525), np.uint8)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            _predictions[p] = 1
    predictions = _predictions

    # predictions = draw_convex_hull(predictions, mode='convex')

    # predictions[test_image <= 2 / 255.] = 0

    if np.sum(predictions) / (350*525) < min_coverage:
        predictions[:,:] = 0

    # dilation
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # RECT, ELLIPSE, CROSS
    # predictions = cv2.dilate(predictions, kernel, iterations=1)

    return predictions

################################

pred = np.load('validation/symmetric_shiftscale/final.npy')
answer = np.load('validation/val_5fold_0.npy')

threshold = [0.6,0.6,0.6,0.6]
min_size = [200,500,2000,1000]
# min_coverage = [0,0,0,0]
min_coverage = [0.04,0.04,0.04,0.02]
fill_up = False

################################

fold_df = pd.read_csv('data/folds/5fold_0.csv', engine='python')
fold_df = fold_df.loc[fold_df['split'] == 'val'].reset_index(drop=True)
img_names = fold_df['Image'].values


classes = ['Fish', 'Flower', 'Gravel', 'Sugar']
for i in range(4):
    for idx in tqdm(range(answer.shape[0])):
        img = cv2.imread(os.path.join('data/train_images', img_names[idx]))
        img = cv2.resize(img, (525, 350))

        _img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        _answer = answer[idx,i] * 255
        cv2.imshow('img', img)
        cv2.putText(_answer, classes[i], (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, 255, 1)
        cv2.imshow('answer', _answer)
        cv2.imshow('pred', np.uint8(cv2.threshold(pred[idx,i,:,:], 0.5, 1, cv2.THRESH_BINARY)[1] * 255))
        cv2.imshow('pred_processed', post_process(pred[idx,i,:,:], _img, threshold[i], min_size[i], min_coverage[i], fill_up) * 255)
        cv2.waitKey()
