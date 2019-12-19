import os
import random
import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from segmentation_models_pytorch import Unet, FPN


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

########################################################################################################################


def get_model(config):
    model_architecture = config.ARCHITECTURE
    model_encoder = config.ENCODER

    # activation은 eval 모드일 때 적용해 주는 거라 train 때에는 직접 sigmoid 쳐야한다.
    if model_architecture == 'Unet':
        model = Unet(model_encoder, encoder_weights='imagenet', classes=4, attention_type='scse')
    elif model_architecture == 'FPN':
        model = FPN(model_encoder, encoder_weights='imagenet', classes=4)

    print('architecture:', model_architecture, 'encoder:', model_encoder)

    return model


########################################################################################################################

class CloudDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform

        fold_df = pd.read_csv(self.config.FOLD_DF, engine='python')
        self.fold_df = fold_df.loc[fold_df['split'] == 'val'].reset_index(drop=True)

        print('len dataset: %s' % len(self.fold_df))

    def __len__(self):
        return len(self.fold_df)

    def __getitem__(self, idx):
        ImageId = self.fold_df["ImageId"][idx]
        image = cv2.imread(os.path.join(self.config.DATA_DIR, ImageId), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config.IMG_W, self.config.IMG_H))

        if self.transform is not None:
            image = self.transform(image)

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        image = normalize(image)

        return image


def get_dataloader(config, transform=None):
    dataset = CloudDataset(config, transform)

    dataloader = DataLoader(dataset,
                             shuffle=False,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_WORKERS,
                             pin_memory=True)

    return dataloader


########################################################################################################################


class VFlip:
    def __call__(self, image):
        return np.flip(image, axis=0).copy()
        # return image[::-1]


class HFlip:
    def __call__(self, image):
        return np.flip(image, axis=1).copy()
        # return image[:,::-1]


########################################################################################################################

def inference(model, dataloader):
    model.eval()

    output = []
    with torch.no_grad():
        start = time.time()
        for i, images in enumerate(dataloader):
            images = images.cuda()
            logits = model(images)
            logits = F.sigmoid(logits)

            preds = logits.detach().cpu().numpy()

            output.append(preds)

            del images, logits, preds
            torch.cuda.empty_cache()

            end = time.time()
            if i % 1 == 0:
                print('[%2d/%2d] time: %.2f' % (i, len(dataloader), end - start))

    output = np.concatenate(output, axis=0)
    print('inference finished. shape:', output.shape)
    return output


def run(config):
    model = get_model(config).cuda()
    checkpoint = torch.load(config.CHECKPOINT)

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    ####################################################################################################
    test_loader = get_dataloader(config, transform=None)
    out = inference(model, test_loader)

    # TTA
    ####################################################################################################
    print('----- VFlip TTA -----')
    test_loader = get_dataloader(config, transform=VFlip())
    out_vflip = inference(model, test_loader)
    out_vflip = np.flip(out_vflip, axis=2)
    out += out_vflip
    del out_vflip
    ####################################################################################################
    print('----- HFlip TTA -----')
    test_loader = get_dataloader(config, transform=HFlip())
    out_hflip = inference(model, test_loader)
    out_hflip = np.flip(out_hflip, axis=3)
    out += out_hflip
    del out_hflip
    ####################################################################################################
    # print('----- HFlip + VFlip TTA -----')
    # test_loader = get_dataloader(config, transform=transforms.Compose([HFlip(),
    #                                                                    VFlip()]))
    # out_vhflip = inference(model, test_loader)
    # out_vhflip = np.flip(out_vhflip, axis=(2,3))
    # out += out_vhflip
    # del out_vhflip
    ####################################################################################################

    out = out / 3.0

    return out


def dice_coef_numpy(preds, labels):
    smooth = 1e-6
    intersection = np.sum(np.float32(preds) * np.float32(labels), axis=(2, 3))
    union = np.sum(np.float32(preds), axis=(2,3)) + np.sum(np.float32(labels), axis=(2,3))
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


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    import warnings
    warnings.filterwarnings("ignore")
    seed_everything()

    ###################### seg #########################

    answer = np.load('validation/answer_fold1.npy')
    # answer = np.load('validation/answer_seg_fold3.npy')

    fold_df_path = 'data/folds/5fold_1_classids.csv'
    # fold_df_path = 'data/folds/seg_5fold_3.csv'

    out_df = pd.DataFrame(columns=['checkpoint', 'val_score'])
    out_df_name = 'Unet_eff-b5_fold1_stratified.csv'

    checkpoint_dir = '_results/Unet_eff-b5_fold1_stratified/checkpoints'
    checkpoints = [os.path.join(checkpoint_dir, checkpoint) for checkpoint in os.listdir(checkpoint_dir)[-15:]]

    for k, checkpoint in enumerate(checkpoints):
        pred = run(Config(architecture='Unet', encoder='efficientnet-b5', fold_df=fold_df_path,
                          checkpoint=checkpoint))

        ####################################################

        threshold = [0.4, 0.4, 0.4, 0.4]
        fill_up = False
        convex = False
        min_size = [4000, 4000, 4000, 4000]
        black_test_image = True
        min_coverage = [0.05, 0.05, 0.05, 0.05]

        ####################################################

        fold_df = pd.read_csv(fold_df_path, engine='python')
        fold_df = fold_df.loc[fold_df['split'] == 'val'].reset_index(drop=True)
        img_names = fold_df['ImageId'].values

        pred_processed = np.zeros((answer.shape[0], 4, 350, 525), np.uint8)

        for idx in tqdm(range(answer.shape[0])):

            val_image = cv2.imread(os.path.join('data/train_images', img_names[idx]), 0)
            val_image = cv2.resize(val_image, (525, 350))
            # val_image = np.float32(val_image)

            for i in range(4):
                pred_processed[idx, i, :, :] = post_process(pred[idx, i, :, :], val_image, threshold[i], min_size[i],
                                                            min_coverage[i], fill_up, convex, black_test_image)

        dice = dice_coef_numpy(pred_processed, answer)
        dice_total = np.mean(dice)
        print(dice)
        print(dice_total)
        out_df.loc[k] = [checkpoint, dice_total]
        print(out_df)

    out_df.to_csv(out_df_name, index=False)


class Config():
    def __init__(self, architecture=None, encoder=None, fold_df=None, checkpoint=None):
        self.ARCHITECTURE = architecture
        self.ENCODER = encoder
        self.CHECKPOINT = checkpoint

        self.IMG_H = 384
        self.IMG_W = 576

        self.DATA_DIR = 'data/train_images'
        # self.FOLD_DF = 'data/folds/seg_5fold_3.csv'
        self.FOLD_DF = fold_df

        self.BATCH_SIZE = 24
        self.NUM_WORKERS = 8


if __name__ == '__main__':
    start = time.time()
    main()
    ellapsed = time.time() - start
    print('Total inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))
