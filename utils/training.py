# import torch

from data import dataloaders
# , resnet_ssd
from architectures.models import SSDLite
from train import optimizer_handler
from general_config import constants, anchor_config, classes_config, general_config
from train.lr_policies import poly_lr, retina_decay

import mindspore as ms
import numpy as np
from mindspore import ops,tensor
import mindspore.nn as nn
import pickle


def update_tensorboard_graphs(writer, loc_loss_train, class_loss_train,
                              loc_loss_val, class_loss_val, mAP, epoch):
    writer.add_scalar('Localization Loss/train', loc_loss_train, epoch)
    writer.add_scalar('Classification Loss/train', class_loss_train, epoch)
    writer.add_scalar('Localization Loss/val', loc_loss_val, epoch)
    writer.add_scalar('Classification Loss/val', class_loss_val, epoch)
    writer.add_scalar('mAP', mAP, epoch)


def gradient_weight_check(model):
    '''
    will pring mean abs value of gradients and weights during training to check for stability
    '''
    avg_grads, max_grads = [], []
    avg_weigths, max_weigths = [], []

    # try to understand comp graph better for why inter variables don't have grad retained and what this means for this stat
    for n, p in model.named_parameters():
        if (p.requires_grad) and not isinstance(p.grad, type(None)):
            avg_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            avg_weigths.append(p.abs().mean())
            max_weigths.append(p.abs().max())

    avg_grads, max_grads = tensor(np.array(avg_grads),dtype=ms.float32),tensor(np.array(max_grads),dtype=ms.float32)

    avg_weigths, max_weigths = tensor(np.array(avg_weigths),dtype=ms.float32),tensor(np.array(max_weigths),dtype=ms.float32)

    return ops.mean(avg_grads), ops.mean(max_grads), ops.mean(avg_weigths), ops.mean(max_weigths)


def model_setup(params):
    """
    creates model and moves it on to cpu/gpu
    """
    n_classes = len(classes_config.training_ids)
    if general_config.model_id == constants.ssdlite:
        model = SSDLite.SSD_Head(n_classes=n_classes, k_list=anchor_config.k_list)
    elif general_config.model_id == constants.ssd:
        model = resnet_ssd.SSD300(n_classes=n_classes)
    elif general_config.model_id == constants.ssd_modified:
        model = SSDLite.SSD_Head(n_classes=n_classes, k_list=anchor_config.k_list,
                                 out_channels=params.out_channels, width_mult=params.width_mult)
    # model.to(general_config.device)

    return model


def optimizer_setup(model, params):
    """
    creates optimizer, can have layer specific options
    """
    if params.optimizer == 'adam':
        if params.freeze_backbone:
            optimizer = optimizer_handler.layer_specific_adam(model, params)
        else:
            optimizer = optimizer_handler.plain_adam(model, params)
    elif params.optimizer == 'sgd':
        if params.freeze_backbone:
            optimizer = optimizer_handler.layer_specific_sgd(model, params)
        else:
            optimizer = optimizer_handler.plain_sgd(model, params)

    # if params.zero_bn_bias_decay:
    #     optimizer = zero_wdcay_bn_bias(optimizer)

    return optimizer


def lr_decay_policy_setup(params, optimizer, loader_size=None):
    if params.lr_policy == constants.poly_lr:
        lr_handler = poly_lr.Poly_LR(params=params, optimizer=optimizer, loader_size=loader_size)
    elif params.lr_policy == constants.retina_lr:
        lr_handler = retina_decay.Retina_decay(params=params, optimizer=optimizer)
    return lr_handler


def prepare_datasets(params):
    train_loader, valid_loader = dataloaders.get_dataloaders(params)
    return train_loader, valid_loader


def load_model(model, params, optimizer):
    checkpoint = ms.load_checkpoint(constants.model_path.format(general_config.model_id))
    ms.load_param_into_net(model,checkpoint)
    checkpoint_opt = ms.load_checkpoint(constants.opt_path.format(general_config.model_id))
    ms.load_param_into_net(optimizer,checkpoint_opt)
    with open('epoch.pkl', 'rb') as f:
        epoch = pickle.load(f)
    return model, optimizer,epoch


def load_weigths_only(model, params):
    checkpoint = ms.load_checkpoint(constants.model_path.format(general_config.model_id))
    ms.load_param_into_net(model,checkpoint)
    print('Weigths loaded successfully')

    return model


def save_model(epoch, model, optimizer, params, stats, msg=None, by_loss=False):
    model_path = constants.model_path
    opt_path = constants.opt_path
    with open('epoch.pkl', 'wb') as f:
        pickle.dump(epoch, f)
    if by_loss:
        model_path = constants.model_path_loss
    ms.save_checkpoint(model, model_path.format(general_config.model_id))
    ms.save_checkpoint(optimizer,opt_path.format(general_config.model_id))
    params.save(constants.params_path.format(general_config.model_id))
    stats.save(constants.stats_path.format(general_config.model_id))

    print(msg)


def update_losses(losses, l_loss, c_loss):
    """
    losses[0], losses[1] - losses from batch nr x to batch nr y
    losses[2], losses[3] - losses per whole data_loader (or multiple epochs if validation does not
    happen after each epoch)
    """
    losses[0] += l_loss
    losses[1] += c_loss
    losses[2] += l_loss
    losses[3] += c_loss


def zero_wdcay_bn_bias(optimizer):
    """
    regroups optimizer param groups such that weight decay on batch norm and bias layers is 0
    """
    new_optim = []
    for pg in optimizer.param_groups:
        for param in pg['params']:
            new_group = {}
            new_group['params'] = param
            # copy rest of attributes as they were before
            for k, v in pg.items():
                if k != 'params':
                    new_group[k] = v
            # if bias or BN
            if len(param.shape) == 1:
                new_group['weight_decay'] = 0
            new_optim.append(new_group)

    # construct a similar optimizer and return it
    return nn.SGD(new_optim) if type(optimizer) == nn.SGD else nn.Adam(new_optim)
