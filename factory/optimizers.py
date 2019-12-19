import torch.optim as optim
from factory.radam import RAdam


def adam(parameters, lr, betas=(0.9, 0.999), weight_decay=0,
         amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                      amsgrad=amsgrad)


def sgd(parameters, lr, momentum=0.9, weight_decay=0, nesterov=True, **_):
    return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                     nesterov=nesterov)


def adamw(parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, **_):
    return optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


def radam(parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True, **_):
    return RAdam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, degenerated_to_sgd=degenerated_to_sgd)


def get_optimizer(config, parameters):
    print('optimizer name:', config.OPTIMIZER.NAME)
    f = globals().get(config.OPTIMIZER.NAME)
    if config.OPTIMIZER.PARAMS is None:
        return f(parameters, config.OPTIMIZER.LR)
    else:
        return f(parameters, config.OPTIMIZER.LR, **config.OPTIMIZER.PARAMS)
