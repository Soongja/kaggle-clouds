import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from albumentations import (
    OneOf, Compose,
    Flip, ShiftScaleRotate, GridDistortion, ElasticTransform,
    RandomGamma, RandomContrast, RandomBrightness, RandomBrightnessContrast,
    Blur, MedianBlur, MotionBlur, GaussianBlur,
    CLAHE, IAASharpen, GaussNoise,
    RandomSizedCrop, CropNonEmptyMaskIfExists,
    RandomSunFlare,
    HueSaturationValue, RGBShift)


def strong_aug(p=1.0):
    return Compose([
        Flip(p=0.75),  # ok
        # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        # RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        # OneOf([
        #     Blur(blur_limit=5, p=1.0),
        #     MedianBlur(blur_limit=5, p=1.0),
        #     MotionBlur(p=1.0),
        # ], p=0.2),
        # OneOf([
        #     HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        #     RGBShift(p=1.0)
        # ], p=0.1),
        # GaussNoise(p=0.1),

        # OneOf([
        #     GridDistortion(p=1.0),
        #     ElasticTransform(p=1.0)
        # ], p=0.2),
        # OneOf([
        #     CLAHE(p=1.0),
        #     IAASharpen(p=1.0),
        # ], p=0.2)
    ], p=p)


def aug2(p=1.0):
    return Compose([
        Flip(p=0.75),
        ShiftScaleRotate(rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ], p=p)


def aug3(p=1.0):
    return Compose([
        Flip(p=0.75),
        ShiftScaleRotate(rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ], p=p)


def aug4(p=1.0):
    return Compose([
        Flip(p=0.75),
        ShiftScaleRotate(rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        GaussianBlur(blur_limit=3, p=0.3),
    ], p=p)


def aug5(p=1.0):
    return Compose([
        Flip(p=0.75),
        ShiftScaleRotate(rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    ], p=p)


class Albu():
    def __init__(self, aug):
        if aug == 1:
            self.aug_func = strong_aug()
        elif aug == 2:
            self.aug_func = aug2()
        elif aug == 3:
            self.aug_func = aug3()
        elif aug == 4:
            self.aug_func = aug4()
        elif aug == 5:
            self.aug_func = aug5()

    def __call__(self, image, mask):
        # augmentation = strong_aug()

        data = {"image": image, "mask": mask}
        # augmented = augmentation(**data)
        augmented = self.aug_func(**data)

        image, mask = augmented["image"], augmented["mask"]

        return image, mask


class Albu_test():
    def __call__(self, image, mask):
        augmentation = Compose([
                            # Flip(p=0.75),
                            # RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
                            # ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
                            # RandomSizedCrop(min_max_height=(1000,1400), height=384, width=576, w2h_ratio=1.5, p=1.0)
                            # CropNonEmptyMaskIfExists(height=300, width=500, p=0.5)
                            # Blur(blur_limit=5, p=0.5),
                            # MedianBlur(blur_limit=5, p=0.5),
                            # MotionBlur(p=0.5),
                            # GaussianBlur(blur_limit=5, p=0.5),
                            # GaussNoise(p=0.5),
                            # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                            # RGBShift(p=1.0)
                            # GridDistortion(p=1.0),
                            # ElasticTransform(p=1.0)
                            # CLAHE(p=0.5),
                            # IAASharpen(p=0.5)
                            RandomSunFlare(p=0.5)
                        ], p=1.0)

        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)

        image, mask = augmented["image"], augmented["mask"]

        return image, mask


if __name__ == "__main__":
    aug = Albu_test()
    # aug = Albu(3)

    img = cv2.imread('00a0954.jpg', 1)
    mask = cv2.imread('00a0954_Gravel.png', 0)

    img = cv2.resize(img, (576, 384))
    mask = cv2.resize(mask, (576, 384), interpolation=cv2.INTER_NEAREST)

    for i in range(100):
        out_img, out_mask = aug(img, mask)

        cv2.imshow('img', cv2.resize(out_img, (576, 384)))
        cv2.imshow('mask', cv2.resize(out_mask, (576, 384)))
        cv2.waitKey()
