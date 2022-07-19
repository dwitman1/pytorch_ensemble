import torch
import numpy as np
import argparse
from torchensemble.normalizers import TorchStandardScaler, TorchInverseStandardScaler
from torchensemble.ensemble import EnsembleRegressor
from torchensemble.architectures import FFNet

def get_data(N):
    x = torch.rand(N,2)
    x[:,0] = 10*x[:,0] - 5
    x[:,1] = 1e3*x[:,1]
    y = torch.zeros(N,2)
    y[:,0] = x[:,0]*torch.sin(.1*x[:,1])
    y[:,1] = x[:,0]*torch.cos(.05*x[:,1])
    return x, y

def main(config):

    # Problem data
    n_features = 2
    n_outputs = 2

    # Network construction
    nodes_per_layer = [(20,30), (30,40)]   
    layers_per_networks = [3, 4]
    networks_per_output = [3, 2]

    # Create the data
    n_train = 5000
    x_train, y_train = get_data(n_train)
    n_val = 100
    x_val, y_val = get_data(n_val)

    # Fit the input/output data
    feature_normalizer = TorchStandardScaler().fit(x_train)
    output_normalizers = [TorchInverseStandardScaler().fit(y_train[:,[i_output]]) for i_output in range(n_outputs)]

    # randomize the hidden sizes for the networks
    hidden_sizes = []
    for i_output in range(n_outputs):
        hidden_sizes.append([[np.random.randint(nodes_per_layer[i_output][0],
                                                nodes_per_layer[i_output][1]) for _ in range(layers_per_networks[i_output])] \
                                                                              for __ in range(networks_per_output[i_output])])

    # Create the networks
    networks = []
    for i_output in range(n_outputs):
        networks_output = []
        for i_network in range(networks_per_output[i_output]):
            networks_output.append(FFNet(n_features, hidden_sizes[i_output][i_network], 1, feature_normalizer=feature_normalizer,
                                         output_normalizer=output_normalizers[i_output]))
        networks.append(networks_output)
    
    # And create the Ensemble
    ensemble = EnsembleRegressor(n_features, networks, n_outputs)

    # Fit it
    ensemble.fit(x_train, y_train, val_data=(x_val, y_val), **config)

    # print the val error
    y_pred_val = ensemble(x_val).detach()
    print(ensemble.error(y_pred_val, y_val))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number oftraining epochs', required=False)
    parser.add_argument('--batch_size', type=int, default=32, help='size of each training batch', required=False)
    parser.add_argument('--loss_type', type=str, default='MSE', 
                        help='Loss function to use (MSE, L1, CrossEntropy)', required=False)
    parser.add_argument('--optimizer_type', type=str, default='Adam', 
                        help='optimizer algorithm to use (Adam, SGD)')
    parser.add_argument('--lr', type=int, default=0.001, help='The learning rate for the optimizer', required=False)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        help='The learning rate scheduler to use (StepLR)', required=False)
    parser.add_argument('--StepLR_step_size', type=int, default=4,
                        help='Number of epochs between learning rate decreases', required=False)
    parser.add_argument('--StepLR_gamma', type=float, default=.1,
                        help='The amount to decrease the learning rate by every StepLR_step_size', required=False)
    parser.add_argument('--sample_strategy', type=str, default='uniform',
                        help='How to drawn training data points from the training dataset', required=False)
    
    args = vars(parser.parse_args())

    print(args)

    main(args)
