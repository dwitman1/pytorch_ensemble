import torch
import numpy as np
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
    layers_per_networks = [4, 5]
    networks_per_output = [4, 2]

    # Create the data
    n_train = 10000
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
    y_pred_val = ensemble(x_val)
    print(ensemble.error(y_pred_val, y_val))
    

if __name__ == '__main__':
    main({})
