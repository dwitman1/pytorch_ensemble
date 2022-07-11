import torch

def get_loss(config):

    if 'loss_type' not in config.keys():
        raise Exception('key: "loss" must be in training parameters')

    if config['loss_type'] == 'MSE':
        return torch.nn.MSELoss()
    elif config['loss_type'] == 'L1':
        return torch.nn.L1Loss()
    elif config['loss_type'] == 'CrossEntropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise Exception('Unrecognized loss function: '+str(config['loss']))

def get_optimizer(config, params):

    if 'optimizer_type' not in config.keys():
        raise Exception('key: "optimizer_type" must be in training parameters')

    if config['optimizer_type'] == 'Adam':
        return torch.optim.Adam(params, )