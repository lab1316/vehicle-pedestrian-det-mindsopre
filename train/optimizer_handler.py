# import torch.optim as optim

"""
Various optimizer setups
"""
import mindspore.nn as nn

def layer_specific_adam(model, params):
    print("AMS grad is false")
    return nn.Adam([
        {'params': model.backbone.trainable_params(), 'lr': params.learning_rate * params.decay_rate},
        {'params': model.loc.trainable_params()},
        {'params': model.conf.trainable_params()},
        {'params': model.additional_blocks.trainable_params()}
    ], learning_rate=params.learning_rate, weight_decay=params.weight_decay, use_amsgrad=False)


def layer_specific_sgd(model, params):
    return nn.SGD([
        {'params': model.backbone.trainable_params(), 'lr': params.learning_rate * params.decay_rate},
        {'params': model.loc.trainable_params()},
        {'params': model.conf.trainable_params()},
        {'params': model.additional_blocks.trainable_params()}
    ], learning_rate=params.learning_rate, weight_decay=params.weight_decay, momentum=0.9)


def plain_adam(model, params):
    return nn.Adam(model.trainable_params(), learning_rate=params.learning_rate, weight_decay=params.weight_decay)


def plain_sgd(model, params):
    return nn.SGD(model.trainable_params(), learning_rate=params.learning_rate,
                     weight_decay=params.weight_decay, momentum=0.9)
