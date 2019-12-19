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
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from utils.rle import mask2rle

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

        self.sample_submission = pd.read_csv(self.config.SAMPLE_SUBMISSION)
        self.ImageIds = np.unique(self.sample_submission['Image_Label'].apply(lambda x: x.split('_')[0]).values)

        print('len dataset: %s' % len(self.ImageIds))

    def __len__(self):
        return len(self.ImageIds)

    def __getitem__(self, idx):
        ImageId = self.ImageIds[idx]
        image = cv2.imread(os.path.join(self.config.DATA_DIR, ImageId), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (self.config.IMG_W, self.config.IMG_H))

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
            if i % 10 == 0:
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
    # print('----- VFlip + HFlip TTA -----')
    # test_loader = get_dataloader(config, transform=transforms.Compose([VFlip(),
    #                                                                    HFlip()]))
    # out_vhflip = inference(model, test_loader)
    # out_vhflip = np.flip(out_vhflip, axis=(2,3))
    # out += out_vhflip
    # del out_vhflip
    ####################################################################################################

    out = out / 3.0

    return out


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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

    threshold = [0.5,0.5,0.5,0.5]

    fill_up = False
    convex = True

    # min_size = [300,1400,3300,2400]
    min_size = [200,500,2000,1000]
    # min_size = [1000,2000,2000,3000]

    black_test_image = True

    # min_coverage = [0,0,0,0]
    min_coverage = [0.04,0.04,0.04,0.02]

    postfix = '191111b'

    ###################### seg #########################

    # fold_0 = run(Config(architecture='Unet', encoder='efficientnet-b3',
    #                     checkpoint='_results/Unet_eff-b3_fold0_symmetric_shiftscale_smoothing1e-4/checkpoints/epoch_0028_score0.6581_loss0.5210.pth'))

    # fold_1 = run(Config(architecture='Unet', encoder='efficientnet-b3',
    #                     checkpoint='_results/Unet_eff-b3_fold1_symmetric_shiftscale_smoothing1e-4/checkpoints/epoch_0023_score0.6522_loss0.5276.pth'))
    # fold_0 += fold_1
    # del fold_1

    # fold_2 = run(Config(architecture='Unet', encoder='efficientnet-b5',
    #                     checkpoint='_results/Unet_eff-b5_fold0_symmetric_shiftscale_smoothing1e-4/checkpoints/epoch_0029_score0.6564_loss0.5204.pth'))
    # fold_0 += fold_2
    # del fold_2

    # fold_3 = run(Config(architecture='Unet', encoder='efficientnet-b5',
    #                     checkpoint='_results/Unet_eff-b5_fold1_symmetric_shiftscale_smoothing1e-4/checkpoints/epoch_0027_score0.6584_loss0.5267.pth'))
    # fold_0 += fold_3
    # del fold_3

    fold = run(Config(architecture='Unet', encoder='efficientnet-b4', data_dir='data/test_images_384_576',
                        checkpoint='_results/Unet_eff-b4_seg_fold3/epoch_0024_score0.6626_loss0.5186.pth'))

    # final = fold_0 / 4.0
    # np.save('final_' + postfix + '.npy', final)

    np.save('npys/Unet_eff-b4_seg_fold3_384_576_0.6626.npy', fold)
    print('success!')

    ####################################################
    '''
    # final = np.load('final_191111a.npy')

    os.makedirs('submissions', exist_ok=True)

    submission['EncodedPixels'] = ''

    test_images = os.listdir('data/test_images')

    for idx in tqdm(range(final.shape[0])):
        test_image = cv2.imread(os.path.join('data/test_images', test_images[idx]), 0)
        test_image = cv2.resize(test_image, (525, 350))

        preds = [mask2rle(post_process(final[idx][i], test_image, threshold=threshold[i], min_size=min_size[i],
                                       min_coverage=min_coverage[i], fill_up=fill_up, convex=convex, black_test_image=black_test_image)) for i in range(4)]

        submission.loc[submission['Image_Label'].apply(lambda x: x.split('_')[0]) == ImageIds[idx], 'EncodedPixels'] = preds

    submission.to_csv(os.path.join('submissions', 'submission_' + postfix + '.csv'), index=False)
    print('success!')
    '''

class Config():
    def __init__(self, architecture=None, encoder=None, data_dir=None, checkpoint=None):
        self.ARCHITECTURE = architecture
        self.ENCODER = encoder
        self.DATA_DIR = data_dir
        self.CHECKPOINT = checkpoint

        self.SAMPLE_SUBMISSION = 'data/sample_submission.csv'

        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 4


if __name__ == '__main__':
    start = time.time()
    main()
    ellapsed = time.time() - start
    print('Total inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))
