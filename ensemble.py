import os
import random
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

from utils.rle import mask2rle


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


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


def change_final_shape(input, out_hw=(350,525)):
    # input shape: [N,C,H,W]
    output = np.zeros((input.shape[0], input.shape[1], out_hw[0], out_hw[1]), np.float32)

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            output[i,j] = cv2.resize(input[i,j], (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)

    return output


def main():
    import warnings
    warnings.filterwarnings("ignore")
    seed_everything()

    submission = pd.read_csv('data/sample_submission.csv', engine='python')
    ImageIds = np.unique(submission['Image_Label'].apply(lambda x: x.split('_')[0]).values)

    ############################################

    cls_threshold = [0,0,0,0]
    # cls_threshold = [0.5,0.5,0.5,0.5]

    # threshold = [0.35,0.375,0.4,0.35]
    # threshold = [0.35,0.4,0.4,0.4]
    # threshold = [0.325,0.4,0.4,0.35]
    threshold = [0.35,0.4,0.4,0.45]

    fill_up = False
    convex = False

    min_size = [4000,4000,4000,4000]

    black_test_image = True

    # min_coverage = [0.07,0.07,0.08,0.04]
    # min_coverage = [0.08,0.07,0.08,0.08]
    min_coverage = [0.08,0.07,0.08,0.08]

    postfix = '191118f'

    ####################################################

    ensembles = [
        'npys/Unet_eff-b4_fold1_384_576_0.6633.npy',
        'npys/Unet_eff-b5_fold1_384_576_0.6648.npy',
        'npys/Unet_inceptionresnetv2_fold1_384_576_0.6624.npy',
        'npys/Unet_se_resnext50_32x4d_fold1_384_576_0.6624.npy',

        'npys/b6_fold3_epoch_0039_score0.6625_loss0.5193.npy',
        'npys/Unet_se_resnext101_32x4d_seg_fold3_384_576_0.6631.npy',
        'npys/b7_fold3_epoch_0036_score0.6634_loss0.5151.npy',
        'npys/Unet_eff-b4_seg_fold3_384_576_0.6626.npy',
                 ]

    ####################################################

    final = np.load(ensembles[0])
    print('0', ensembles[0])

    for i in range(1, len(ensembles)):
        print(i, ensembles[i])
        fold = np.load(ensembles[i])
        if fold.shape[2] == 320:
            fold = change_final_shape(fold, (384,576))
        final += fold
        del fold

    final = final / (float(len(ensembles)))

    ####################################################

    os.makedirs('submissions', exist_ok=True)

    submission['EncodedPixels'] = ''

    test_images = os.listdir('data/test_images')

    max_vals = []
    for idx in tqdm(range(final.shape[0])):
        test_image = cv2.imread(os.path.join('data/test_images', test_images[idx]), 0)
        test_image = cv2.resize(test_image, (525, 350))

        preds = []
        for i in range(4):
            max_vals.append(np.max(final[idx,i]))
            if np.max(final[idx,i]) < cls_threshold[i]:
                preds.append('')
                print('oh no!!')
            else:
                preds.append(mask2rle(post_process(final[idx,i], test_image, threshold=threshold[i], min_size=min_size[i],
                                       min_coverage=min_coverage[i], fill_up=fill_up, convex=convex, black_test_image=black_test_image)))

        # preds = [mask2rle(post_process(final[idx][i], test_image, threshold=threshold[i], min_size=min_size[i],
        #                                min_coverage=min_coverage[i], fill_up=fill_up, convex=convex, black_test_image=black_test_image)) for i in range(4)]

        submission.loc[submission['Image_Label'].apply(lambda x: x.split('_')[0]) == ImageIds[idx], 'EncodedPixels'] = preds

    submission.to_csv(os.path.join('submissions', 'submission_' + postfix + '.csv'), index=False)
    print('success!')

    print(max_vals)


if __name__ == '__main__':
    start = time.time()
    main()
    ellapsed = time.time() - start
    print('Total inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))
