import torch


def dice_coef(preds, labels):
    preds = preds >= 0.5
    labels = labels >= 0.5

    smooth = 1e-6
    intersection = (preds.float() * labels.float()).sum(dim=(2, 3))
    union = preds.float().sum(dim=(2, 3)) + labels.float().sum(dim=(2, 3))
    # class 별로 찍게 하자 [N, C]
    dice = ((2. * intersection + smooth) / (union + smooth)).mean(dim=0)

    return dice


if __name__ == '__main__':
    import time

    preds = torch.rand(8, 4, 128, 128)
    labels = torch.randint(0, 2, (8, 4, 128, 128))

    start = time.time()
    for i in range(1000):
        dice = dice_coef(preds, labels)
    print(time.time() - start)
