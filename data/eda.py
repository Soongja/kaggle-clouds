import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a'
        self.file = open(file, mode)

    def write(self, message, is_terminal=0, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# log = Logger()
# log.open('eda.txt')


df = pd.DataFrame(columns=['fname', 'hasMask', 'numComponents', 'minComponentArea', 'coverage'])


cnt = 0
classes = ['Fish', 'Flower', 'Gravel', 'Sugar']
for cls in classes:
    fnames = [fname for fname in os.listdir('train_masks') if fname.endswith(cls + '.png')]
    print(fnames)

    # total /= len(fnames)
    # missing = AverageMeter()
    # nonmissing = AverageMeter()
    # min_area = float('inf')
    # max_area = -1
    # coverage = AverageMeter()

    for fname in tqdm(fnames):
        mask = cv2.imread(os.path.join('train_masks', fname), 0)

        mask2 = cv2.medianBlur(mask, ksize=11)
        mask2 = cv2.medianBlur(mask2, ksize=11)
        _, mask2 = cv2.threshold(mask2, 127, 255, 0)

        mask = mask / 255

        if np.sum(mask) == 0:
            hasmask = 0
            # missing.update(1)
        elif np.sum(mask) != 0:
            hasmask = 1
            # nonmissing.update(1)

        # if np.sum(mask) > max_area:
        #     max_area = np.sum(mask)
        # if np.sum(mask) != 0 and np.sum(mask) < min_area:
        #     min_area = np.sum(mask)

        coverage = np.sum(mask) / (1400*2100)

        # if np.sum(mask) != 0:
        #     coverage.update(np.sum(mask) / (1400*2100))

        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        num_component, component = cv2.connectedComponents(mask2.astype(np.uint8))
        # print(num_component)


        mean_area = 0

        if num_component == 1:
            min_area = 0
        else:
            min_area = 1400*2100
            for c in range(1, num_component):
                p = (component == c)
                area = p.sum()
                # print(p.sum())
                if area < min_area:
                    min_area = area
                mean_area += area
            mean_area = int(mean_area / (num_component - 1))

        df.loc[cnt] = [fname, hasmask, num_component-1, min_area, coverage]
        # df.append([fname, hasmask, num_component, min_area, coverage])
        # print(df)
        cnt += 1

df.to_csv('eda.csv', index=False)
    # log.write('[%s] total: %d, missing: %d, nonmissing: %d, min_area: %d, max_area: %d, coverage: %.2f\n'
    #           % (cls, total, missing.sum, nonmissing.sum, min_area, max_area, coverage.avg))
