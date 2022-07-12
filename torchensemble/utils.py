import torch
from copy import deepcopy

defaults = {}
#
# Optimizer Defaults
defaults['Adam_lr'] = 0.001
defaults['Adam_L2'] = 1e-5
defaults['Adam_betas'] =(0.9, 0.999)
defaults['Adam_eps'] = 1e-08
defaults['SGD_lr'] = .001
defaults['SGD_momentum'] = 0.8
defaults['SGD_dampening'] = 0
defaults['SGD_weight_decay'] = 1e-5

#
# Scheduler Defaults
defaults['StepLR_step_size'] = 5
defaults['StepLR_gamma'] = .1

#
# Dataloader defaults
defaults['uniform']
defaults['low_prob_scale'] = 1
defaults['replacement'] = True

def _sync_config_with_defaults(config):
    config = deepcopy(config)
    for k in defaults:
        config[k] = defaults[k]
    return config

def get_loss(config):

    # Sync with any defaults that might exist
    config =_sync_config_with_defaults(config)

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

    # Sync with any defaults that might exist
    config =_sync_config_with_defaults(config)

    if 'optimizer_type' not in config.keys():
        raise Exception('key: "optimizer_type" must be in training parameters')

    if config['optimizer_type'] == 'Adam':
        # Note: Adam's weight decay in pytorch is weird and is actually L2 regularization
        # https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
        return torch.optim.Adam(params, lr=config['Adam_lr'], betas=config['Adam_betas'],
                                eps=config['Adam_eps'], weight_decay=config['Adam_L2'])
    elif config['optimizer_type'] == 'SGD':
        return torch.optim.SGD(params, lr=config['SGD_lr'], momentum=config['SGD_momentum'],
                               dampening=config['SGD_dampening'], weight_decay=config['SGD_weight_decay'])
    else:
        raise Exception('Optimizer type: '+str(config['optimizer_type'])+' not supported')

def get_lr_scheduler(config, optimizer):

    # Sync with any defaults that might exist
    config =_sync_config_with_defaults(config)

    # We dont require a lr scheduler
    if 'lr_scheduler' not in config.keys():
        return None
    
    if config['lr_scheduler'] == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size= config['StepLR_step_size'],
                                               gamma=config['StepLR_gamma'])
    else:
        raise Exception('Scheduler type: '+str(config['lr_scheduler'])+' not supported')

def get_dataloader(config, x_train, y_train):

    # Sync with any defaults that might exist
    config =_sync_config_with_defaults(config)

    # Create the torch dataset
    tensor_dataset = torch.utils.data.TensorDataset(x_train, y_train)

    # If it isnt defined, default to uniform
    if 'sample_strategy' not in config.keys():
        config['sample_strategy'] = 'uniform'

    # Get the number of training instances
    n_train = x_train.shape[0]

    if config['sample_strategy'] == 'uniform':
        w = torch.ones(n_train)
    elif config['sample_strategy'] == 'low_prob':
        w = torch.sum(torch.abs(y_train - torch.tile(torch.mean(y_train, dim=0), (n_train,1))), dim=1).pow(defaults['low_prob_scale'])
    else:
        raise Exception('Unrecognized sample strategy: '+str(config['sample_strategy']))

    # Create random sampler
    w_sampler = torch.utils.data.WeightedRandomSampler(w,n_train,replacement=config['replacement'])
    
    # Create the data loader
    dl = torch.utils.data.DataLoader(tensor_dataset,batch_size=config['batch_size'],
                                     sampler=w_sampler,drop_last=True)
    
    return dl