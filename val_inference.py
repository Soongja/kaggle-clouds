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


def run(config, tta=True):
    model = get_model(config).cuda()
    checkpoint = torch.load(config.CHECKPOINT)

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    ####################################################################################################
    test_loader = get_dataloader(config, transform=None)
    out = inference(model, test_loader)

    if tta:
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

    # final = run(Config(architecture='Unet', encoder='efficientnet-b4', fold_df='data/folds/5fold_1_classids.csv',
    #                   checkpoint='_results/Unet_eff-b4_fold1_sj/epoch_0028_score0.6633_loss0.5135.pth'))
    #
    # fold_0 = run(Config(architecture='Unet', encoder='efficientnet-b5', fold_df='data/folds/5fold_1_classids.csv',
    #                   checkpoint='_results/Unet_eff-b5_fold1_stratified/epoch_0023_score0.6648_loss0.5130.pth'))
    # final += fold_0
    # del fold_0
    #
    # fold_1 = run(Config(architecture='Unet', encoder='inceptionresnetv2', fold_df='data/folds/5fold_1_classids.csv',
    #                     checkpoint='_results/Unet_inceptionresnetv2_fold1_stratified/epoch_0016_score0.6624_loss0.5146.pth'))
    # final += fold_1
    # del fold_1
    #
    # fold_2 = run(Config(architecture='Unet', encoder='se_resnext50_32x4d', fold_df='data/folds/5fold_1_classids.csv',
    #                     checkpoint='_results/Unet_se_resnext50_32x4d_fold1_stratified/epoch_0026_score0.6624_loss0.5155.pth'))
    # final += fold_2
    # del fold_2

    final = run(Config(architecture='Unet', encoder='efficientnet-b6', fold_df='data/folds/seg_5fold_3.csv',
                       checkpoint='_results/sejun-b6-seg_fold3/epoch_0039_score0.6625_loss0.5193.pth'))

    fold_0 = run(Config(architecture='Unet', encoder='se_resnext101_32x4d', fold_df='data/folds/seg_5fold_3.csv',
                        checkpoint='_results/Unet_se_resnext101_32x4d_seg_fold3/epoch_0027_score0.6631_loss0.5212.pth'))
    final += fold_0
    del fold_0

    fold_1 = run(Config(architecture='Unet', encoder='efficientnet-b7', fold_df='data/folds/seg_5fold_3.csv',
                        checkpoint='_results/sejun-b7-seg_fold3/epoch_0036_score0.6634_loss0.5151.pth'))
    final += fold_1
    del fold_1

    fold_2 = run(Config(architecture='Unet', encoder='efficientnet-b4', fold_df='data/folds/seg_5fold_3.csv',
                        checkpoint='_results/Unet_eff-b4_seg_fold3/epoch_0024_score0.6626_loss0.5186.pth'))
    final += fold_2
    del fold_2

    final = final / 4.0

    # np.save('validation/fold1_swa_b5.npy', final)
    # np.save('validation/fold1_ensemble4_0.6633_0.6648_0.6624_0.6624.npy', final)
    np.save('validation/seg_fold3_ensemble4_0.6625_0.6631_0.6634_0.6626.npy', final)


class Config():
    def __init__(self, architecture=None, encoder=None, fold_df=None, checkpoint=None):
        self.ARCHITECTURE = architecture
        self.ENCODER = encoder
        self.CHECKPOINT = checkpoint

        self.IMG_H = 384
        self.IMG_W = 576

        self.DATA_DIR = 'data/train_images'
        self.FOLD_DF = fold_df
        # self.FOLD_DF = 'data/folds/seg_5fold_3.csv'

        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 4


if __name__ == '__main__':
    start = time.time()
    main()
    ellapsed = time.time() - start
    print('Total inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))
