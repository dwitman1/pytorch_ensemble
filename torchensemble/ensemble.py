import torch

training_parameters = {}
training_parameters['epochs'] = 10
training_parameters['batch_size'] = 32
training_parameters['loss'] = 'MSE'
training_parameters['optimizer_type'] = 'Adam'
training_parameters['lr'] = .001
training_parameters['lr_scheduler'] = 'StepLR'
training_parameters['lr_scheduler_steps2reduce'] = 4
training_parameters['lr_scheduler_gamma'] = .1



class EnsembleRegressor(torch.nn.Module):

    def __init__(self, n_features, networks, n_outputs):

        super(EnsembleRegressor, self).__init__()

        # Set the atributes
        self.n_features = n_features
        self.networks = networks
        self.n_outputs = n_outputs

        # Initialize the weights
        self.weights = [[1/len(networks[i_output]) for _ in range(len(networks[i_output]))] for i_output in range(self.n_outputs)]

    
    def forward(self, x):

        # Define the output
        y = torch.zeros(x.shape[0], self.n_outputs)

        # Loop over all the outputs
        for i_output in range(self.n_outputs):
            # And all the networks
            for i_network in range(len(self.networks[i_output])):
                y[:,i_output] += self.weights[i_output][i_network]*self.networks[i_output][i_network](x)
        
        return y
    
    def fit(self, x_train, y_train, val_data=None, **kwargs):

        # Pass any optimizer parameters to the individual optimizer
        for key in kwargs:
            training_parameters[key] = kwargs[key]

        # If there is no validation data, extract 10% of training data
        if val_data is None:
            idx_perm = torch.randperm(x_train.shape[0])
            idx_val = idx_perm[:int(.1*x_train.shape[0])]
            idx_train = idx_perm[int(.1*x_train.shape[0]):]
            # Extract the validation data from training dataset
            x_val = x_train[idx_val, :]
            y_val = y_train[idx_val, :]
            # Reduce the training data to not include the val data
            x_train = x_train[idx_train, :]
            y_train = y_train[idx_train, :]
        else:
            x_val = val_data[0]
            y_val = val_data[1]
        
        # For all the outputs
        for i_output in range(self.n_outputs):
            # For all the networks
            for i_network in range(len(self.networks)):
                # Fit this network

        