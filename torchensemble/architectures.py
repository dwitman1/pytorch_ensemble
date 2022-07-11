import torch

training_parameters = {}
training_parameters['epochs'] = 10
training_parameters['batch_size'] = 32
training_parameters['loss_type'] = 'MSE'
training_parameters['optimizer_type'] = 'Adam'
training_parameters['lr'] = .001
training_parameters['lr_scheduler'] = 'StepLR'
training_parameters['lr_scheduler_steps2reduce'] = 4
training_parameters['lr_scheduler_gamma'] = .1

class FFNet(torch.nn.Module):

    def __init__(self, n_features, hidden_sizes, n_outputs,
                 activation=torch.tanh(),
                 normalize_features=None, normalize_output=None):

        # Store the attributes
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.n_outputs = n_outputs

        # The activation function
        self.activation = activation
        
        # If we need to do input normalization
        if normalize_features is not None:
            self.layers = [normalize_features]
        else:
            self.layers = []

        # Define the initial layers
        self.layers.append(torch.nn.Linear(n_features, hidden_sizes[0]))
        self.layers.append(self.activation)
        
        # Loop over the layers
        for i_layer in range(1, len(hidden_sizes)):
            self.layers.append(torch.nn.Linear(hidden_sizes[i_layer-1],
                                               hidden_sizes[i_layer]))
            self.layers.append(self.activation)
        
        # Add the final output layer
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], n_outputs))

        # If there is output normalization
        if normalize_output is not None:
            self.layers.append(normalize_output)
        
        # Set the network
        self.network = torch.nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.network(x)

    def get_latent_data(self, x):
        pass
    
    def fit(x_train, y_train, **kwargs):

        # Pass any optimizer parameters to the individual optimizer
        for key in kwargs:
            training_parameters[key] = kwargs[key]
        
        # Set the loss
        





