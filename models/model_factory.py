import torch.nn as nn
from segmentation_models_pytorch import Unet, Linknet, FPN, PSPNet
from .get_csail import get_hrnetv2, get_resnet50_upernet, get_resnet101_upernet
from .acnet import ACNet
from .deeplabv3_pytorch import get_deeplabv3
from .modeling.deeplab import DeepLab


def get_model(config):

    if config.MODEL.NAME == 'hrnetv2':
        model = get_hrnetv2()
        print('model: hrnetv2')

    elif config.MODEL.NAME == 'resnet50_upernet':
        model = get_resnet50_upernet()
        print('model: resnet50_upernet')

    elif config.MODEL.NAME == 'resnet101_upernet':
        model = get_resnet101_upernet()
        print('model: resnet101_upernet')

    elif config.MODEL.NAME == 'acnet':
        model = ACNet(num_class=4, pretrained=True)
        print('model: acnet')

    elif config.MODEL.NAME == 'deeplabv3':
        model = get_deeplabv3()
        print('model: deeplabv3')

    elif config.MODEL.NAME =='deeplab_xception':
        model = DeepLab(backbone='xception', output_stride=16, num_classes=4,
                 sync_bn=False, freeze_bn=False)

    else:
        model_architecture = config.MODEL.ARCHITECTURE
        model_encoder = config.MODEL.ENCODER
        model_pretrained = config.MODEL.PRETRAINED

        if model_architecture == 'Unet':
            model = Unet(model_encoder, encoder_weights=model_pretrained, classes=4, attention_type='scse')
        elif model_architecture == 'Linknet':
            model = Linknet(model_encoder, encoder_weights=model_pretrained, classes=4)
        elif model_architecture == 'FPN' or model_architecture == 'PSPNet':
            model = FPN(model_encoder, encoder_weights=model_pretrained, classes=4)

        print('architecture:', model_architecture, 'encoder:', model_encoder, 'pretrained on:', model_pretrained)

    if config.PARALLEL:
        model = nn.DataParallel(model)

    print('[*] num parameters:', count_parameters(model))

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
