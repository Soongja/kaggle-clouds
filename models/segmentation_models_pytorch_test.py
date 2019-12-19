import torch
from segmentation_models_pytorch import Unet, Linknet, FPN, PSPNet


model = Unet('se_resnet152', encoder_weights='imagenet', classes=4, activation='sigmoid')
# model = Unet('resnext101_32x8d', encoder_weights='instagram', classes=4, activation='sigmoid')
# model = Unet('dpn92', encoder_weights='imagenet+5k', classes=4, activation='sigmoid')
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))

# weights_dict = {
#     'state_dict': model.state_dict(),
# }
# torch.save(weights_dict, 'check.pth')
