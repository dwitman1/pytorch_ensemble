import torch
import numpy as np
from numpy.random import randint
import requests


def get_data(N):
    x = torch.rand(N,2)
    x[:,0] = 10*x[:,0] - 5
    x[:,1] = 1e3*x[:,1]
    y = torch.zeros(N,2)
    y[:,0] = x[:,0]*torch.sin(.1*x[:,1])
    y[:,1] = x[:,0]*torch.cos(.05*x[:,1])
    return x, y

def main():
    config = {}
    n_features = 2
    n_outputs = 2
    n_networks = 4
    n_train = 2500
    n_val = 100
    # Create the number of layers as a random int by n_outputs
    n_layers = [[randint(3,5) for _ in range(n_networks)] for __ in range(n_outputs)]
    # Range of nodes per layer
    nodes_per_layer = (20,30)
    # Put it into the json
    config['name'] = 'test'
    config['n_features'] = 2
    config['n_outputs'] = 2
    # Create a list of ints of size:n_outputs x n_networks x n_hidden_layers
    config['hidden_sizes'] = [[randint(nodes_per_layer[0],
                                    nodes_per_layer[1],
                                    size=(n_layers[i_output][i_network],)).tolist() for i_network in range(n_networks)] for i_output in range(n_outputs)]
    # Normalizations
    # For input features, just have one normalizer for the inputs
    config['feature_normalizer'] = 'TorchStandardScaler'
    # For outputs, create normalizer for each of the outputs
    config['output_normalizer'] = 'TorchInverseStandardScaler'

    # Training data
    x, y = get_data(n_train)
    config['x_train'] = x.tolist()
    config['y_train'] = y.tolist()
    # Validation data
    x, y = get_data(n_val)
    config['x_val'] = x.tolist()
    config['y_val'] = y.tolist()

    # Making a POST request'
    r = requests.post('http://127.0.0.1:8000/train', json=config)
    
    # check status code for response received
    # success code - 200
    print(r)
    
    # print content of request
    print(r.json())

if __name__ == '__main__':
    main()