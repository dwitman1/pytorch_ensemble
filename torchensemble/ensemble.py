import torch
import uuid


class EnsembleRegressor(torch.nn.Module):

    def __init__(self, n_features, networks, n_outputs):

        super(EnsembleRegressor, self).__init__()

        # Set the atributes
        self.n_features = n_features
        self.networks = networks
        self.n_outputs = n_outputs

        # Designate the ensemble unique id
        self.ensemble_id = str(uuid.uuid4())

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
        
        # The validation errors to use to compute the weights
        self.val_errors = [[None for _ in range(len(self.networks[i_output]))] for i_output in range(self.n_outputs)]

        # The training errors for each network
        training_errors = [[None for _ in range(self.networks[i_output])] for i_output in range(self.n_outputs)]

        # For all the outputs
        for i_output in range(self.n_outputs):
            # For all the networks
            for i_network in range(len(self.networks)):
                # Fit this network
                training_errors[i_output][i_network] = self.networks[i_output][i_network].fit(x_train, y_train, kwargs)
                # Calculate the validation error
                y_val_pred = self.networks[i_output][i_network](x_val).detach()
                self.val_errors[i_output][i_network] = self.error(y_val_pred, y_val)
        
        # Compute the weights based on the validation errors
        self.weights = self._compute_weights(self.val_errors)
    

    def error(self, y_pred, y_true):
        return torch.nn.MSELoss()(y_pred, y_true)
    
    def _compute_weights(self, errors, alpha=.05, beta=-1):

        w = [None for _ in range(self.n_outputs)]

        for i_output in range(self.n_outputs):
            errors_output = torch.FloatTensor(errors[i_output])
            # COmpute intermediate weights
            w_star = (errors_output + alpha*torch.mean(errors_output)).pow(beta)
            # And the final weights
            w_output = w_star/torch.sum(w_star)
            # Add it to the list
            w[i_output] = w_output.detach().tolist()
        
        return w


        