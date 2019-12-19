from glob import iglob
import numpy as np
import cv2
from tqdm import tqdm
import os

# As we count the statistics, we can check if there are any completely black or white images
train_dir = "train_images"
test_dir = "test_images"

x_tot = np.zeros(3)
x2_tot = np.zeros(3)


x_tot_test = np.zeros(3)
x2_tot_test = np.zeros(3)
cnt_test = 0

cnt = 0


print("train data search in RGB space")
train_fnames = os.listdir(train_dir)

for fname in tqdm(train_fnames):
    image = cv2.imread(os.path.join(train_dir, fname))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(-1, 3) / 255.

    # train #
    x_tot += image.mean(axis=0)
    x2_tot += (image ** 2).mean(axis=0)
    cnt += 1

channel_avr_train = x_tot / cnt
channel_std_train = np.sqrt(x2_tot / cnt - channel_avr_train ** 2)


print("\ntrain_avr:" +str(channel_avr_train))
print("train_std:" +str(channel_std_train))

print("test data search")
test_fnames = os.listdir(test_dir)

for fname in tqdm(test_fnames):
    image = cv2.imread(os.path.join(test_dir, fname))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(-1, 3) / 255.

    # total_add #
    x_tot += image.mean(axis=0)
    x2_tot += (image ** 2).mean(axis=0)
    cnt += 1
    # test #
    x_tot_test += image.mean(axis=0)
    x2_tot_test += (image ** 2).mean(axis=0)
    cnt_test += 1

channel_avr_test = x_tot_test / cnt_test
channel_std_test = np.sqrt(x2_tot_test / cnt_test - channel_avr_test ** 2)


print("\navr_test:" +str(channel_avr_test))
print("std_test:" +str(channel_std_test))

channel_avr_total = x_tot / cnt
channel_std_total = np.sqrt(x2_tot / cnt - channel_avr_total ** 2)


print("\navr_total:" +str(channel_avr_total))
print("std_total:" +str(channel_std_total))