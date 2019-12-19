import os
import sys
import random
import shutil
import cv2
import time
import math
import pprint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.model_factory import get_model
from factory.losses import get_loss
from factory.schedulers import get_scheduler
from factory.optimizers import get_optimizer
from factory.transforms import Albu
from datasets.dataloader import get_dataloader

import utils.config
import utils.checkpoint
from utils.metrics import dice_coef
from utils.tools import prepare_train_directories, AverageMeter, Logger
from utils.experiments import LabelSmoother


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_single_epoch(config, model, dataloader, criterion, log_val, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    score_1 = AverageMeter()
    score_2 = AverageMeter()
    score_3 = AverageMeter()
    score_4 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # logits = model(images)
            logits = model(images)['out']

            loss = criterion(logits, labels)
            losses.update(loss.item(), images.shape[0])

            preds = F.sigmoid(logits)

            score = dice_coef(preds, labels)
            score_1.update(score[0].item(), images.shape[0])
            score_2.update(score[1].item(), images.shape[0])
            score_3.update(score[2].item(), images.shape[0])
            score_4.update(score[3].item(), images.shape[0])
            scores.update(score.mean().item(), images.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_EVERY == 0:
                print('[%2d/%2d] time: %.2f, loss: %.6f, score: %.4f [%.4f, %.4f, %.4f, %.4f]'
                      % (i, len(dataloader), batch_time.sum, loss.item(), score.mean().item(), score[0].item(), score[1].item(), score[2].item(), score[3].item()))

            del images, labels, logits, preds
            torch.cuda.empty_cache()

        log_val.write('[%d/%d] loss: %.6f, score: %.4f [%.4f, %.4f, %.4f, %.4f]\n'
                      % (epoch, config.TRAIN.NUM_EPOCHS, losses.avg, scores.avg, score_1.avg, score_2.avg, score_3.avg, score_4.avg))
        print('average loss over VAL epoch: %f' % losses.avg)

    return scores.avg, losses.avg


def train_single_epoch(config, model, dataloader, criterion, optimizer, log_train, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    score_1 = AverageMeter()
    score_2 = AverageMeter()
    score_3 = AverageMeter()
    score_4 = AverageMeter()

    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        # logits = model(images)
        logits = model(images)['out']

        if config.LABEL_SMOOTHING:
            smoother = LabelSmoother()
            loss = criterion(logits, smoother(labels))
        else:
            loss = criterion(logits, labels)

        losses.update(loss.item(), images.shape[0])

        loss.backward()
        optimizer.step()

        preds = F.sigmoid(logits)

        score = dice_coef(preds, labels)
        score_1.update(score[0].item(), images.shape[0])
        score_2.update(score[1].item(), images.shape[0])
        score_3.update(score[2].item(), images.shape[0])
        score_4.update(score[3].item(), images.shape[0])
        scores.update(score.mean().item(), images.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_EVERY == 0:
            print("[%d/%d][%d/%d] time: %.2f, loss: %.6f, score: %.4f [%.4f, %.4f, %.4f, %.4f], lr: %.6f"
                  % (epoch, config.TRAIN.NUM_EPOCHS, i, len(dataloader), batch_time.sum, loss.item(), score.mean().item(),
                     score[0].item(), score[1].item(), score[2].item(), score[3].item(),
                     optimizer.param_groups[0]['lr']))
                     # optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))

        del images, labels, logits, preds
        torch.cuda.empty_cache()

    log_train.write('[%d/%d] loss: %.6f, score: %.4f, dice: [%.4f, %.4f, %.4f, %.4f], lr: %.6f\n'
                    % (epoch, config.TRAIN.NUM_EPOCHS, losses.avg, scores.avg, score_1.avg, score_2.avg, score_3.avg, score_4.avg,
                       optimizer.param_groups[0]['lr']))
                       # optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
    print('average loss over TRAIN epoch: %f' % losses.avg)


def train(config, model, train_loader, test_loader, optimizer, scheduler, log_train, log_val, start_epoch, best_score, best_loss):
    num_epochs = config.TRAIN.NUM_EPOCHS
    model = model.to(device)

    for epoch in range(start_epoch, num_epochs):

        if epoch >= config.LOSS.FINETUNE_EPOCH:
            criterion = get_loss(config.LOSS.FINETUNE_LOSS)
        else:
            criterion = get_loss(config.LOSS.NAME)

        train_single_epoch(config, model, train_loader, criterion, optimizer, log_train, epoch)

        test_score, test_loss = evaluate_single_epoch(config, model, test_loader, criterion, log_val, epoch)

        print('Total Test Score: %.4f, Test Loss: %.4f' % (test_score, test_loss))

    #     if test_score > best_score:
    #         best_score = test_score
    #         print('Test score Improved! Save checkpoint')
    #         utils.checkpoint.save_checkpoint(config, model, epoch, test_score, test_loss)

        utils.checkpoint.save_checkpoint(config, model, epoch, test_score, test_loss)

        if config.SCHEDULER.NAME == 'reduce_lr_on_plateau':
            scheduler.step(test_score)
        else:
            scheduler.step()


def run(config):
    model = get_model(config).to(device)
    # model_params = [{'params': model.encoder.parameters(), 'lr': config.OPTIMIZER.ENCODER_LR},
    #                 {'params': model.decoder.parameters(), 'lr': config.OPTIMIZER.DECODER_LR}]

    optimizer = get_optimizer(config, model.parameters())
    # optimizer = get_optimizer(config, model_params)

    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, score, loss = utils.checkpoint.load_checkpoint(config, model, checkpoint)
    else:
        print('[*] no checkpoint found')
        last_epoch, score, loss = -1, -1, float('inf')
    print('last epoch:{} score:{:.4f} loss:{:.4f}'.format(last_epoch, score, loss))

    optimizer.param_groups[0]['initial_lr'] = config.OPTIMIZER.LR
    # optimizer.param_groups[0]['initial_lr'] = config.OPTIMIZER.ENCODER_LR
    # optimizer.param_groups[1]['initial_lr'] = config.OPTIMIZER.DECODER_LR

    scheduler = get_scheduler(config, optimizer, last_epoch)

    if config.SCHEDULER.NAME == 'multi_step':
        milestones = scheduler.state_dict()['milestones']
        step_count = len([i for i in milestones if i < last_epoch])
        optimizer.param_groups[0]['lr'] *= scheduler.state_dict()['gamma'] ** step_count
        # optimizer.param_groups[0]['lr'] *= scheduler.state_dict()['gamma'] ** step_count
        # optimizer.param_groups[1]['lr'] *= scheduler.state_dict()['gamma'] ** step_count

    if last_epoch != -1:
        scheduler.step()

    log_train = Logger()
    log_val = Logger()
    log_train.open(os.path.join(config.TRAIN_DIR, 'log_train.txt'), mode='a')
    log_val.open(os.path.join(config.TRAIN_DIR, 'log_val.txt'), mode='a')

    train_loader = get_dataloader(config, 'train', transform=Albu(config.ALBU))
    val_loader = get_dataloader(config, 'val')

    train(config, model, train_loader, val_loader, optimizer, scheduler, log_train, log_val, last_epoch+1, score, loss)


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

    print('start training.')
    seed_everything()

    # yml = 'configs/' + sys.argv[1]
    # yml = 'configs/seg.yml'

    ymls = ['configs/seg.yml']
    # ymls = ['configs/seg.yml', 'configs/seg2.yml']
    for yml in ymls:
        config = utils.config.load(yml)
        prepare_train_directories(config)
        pprint.pprint(config, indent=2)
        utils.config.save_config(yml, config.TRAIN_DIR)
        run(config)

    print('success!')


if __name__ == '__main__':
    main()
