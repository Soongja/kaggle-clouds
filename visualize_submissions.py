import os
import cv2
import numpy as np
import pandas as pd

from utils.rle import rle2mask


def get_mask(submission, idx):
    rle = submission['EncodedPixels'][idx]

    if str(rle) != 'nan':
        mask = rle2mask(rle, shape=(350, 525))
    else:
        mask = np.zeros((350, 525), np.uint8)

    return mask


submissiond = pd.read_csv('submissions/submission_191116d.csv')
# submissione = pd.read_csv('submissions/submission_191116e.csv')
# submissionb = pd.read_csv('submissions/submission_191116b.csv')
# submissionf = pd.read_csv('submissions/submission_191116f.csv')
# submissiong = pd.read_csv('submissions/submission_191116g.csv')
submission7 = pd.read_csv('submissions/submission_191117test.csv')

print(len(submissiond))
coverages = []
for i in range(len(submissiond)):
    mask = get_mask(submissiond, i) * 255
    # mask2 = get_mask(submissione, i)
    # mask3 = get_mask(submissionb, i)
    # mask4 = get_mask(submissionf, i)
    # mask5 = get_mask(submissiong, i)
    mask7 = get_mask(submission7, i) * 255
    #
    cv2.imshow('mask', mask)
    # cv2.imshow('mask2', mask2)
    # cv2.imshow('mask3', mask3)
    # cv2.imshow('mask4', mask4)
    # cv2.imshow('mask5', mask5)
    cv2.imshow('mask7', mask7)
    cv2.waitKey()

    coverages.append((np.sum(mask) / (350*525)))
coverages.sort()
print(coverages)
