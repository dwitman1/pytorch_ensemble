from matplotlib.pyplot import get
import torch
from utils import get_loss, get_optimizer, get_lr_scheduler, get_dataloader
from tqdm import tqdm
import uuid

training_parameters = {}
training_parameters['epochs'] = 10
training_parameters['batch_size'] = 32
training_parameters['loss_type'] = 'MSE'
training_parameters['optimizer_type'] = 'Adam'
training_parameters['Adam_lr'] = .001
training_parameters['lr_scheduler'] = 'StepLR'
training_parameters['StepLR_step_size'] = 4
training_parameters['StepLR_gamma'] = .1
training_parameters['sample_strategy'] = 'uniform'

class FFNet(torch.nn.Module):

    def __init__(self, n_features, hidden_sizes, n_outputs,
                 activation=torch.tanh(),
                 feature_normalizer=None, output_normalizer=None):

        # Store the attributes
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.n_outputs = n_outputs

        # Designate the unique id
        self.network_id = str(uuid.uuid4())

        # The activation function
        self.activation = activation
        
        # If we need to do input normalization
        if feature_normalizer is not None:
            self.layers = [feature_normalizer]
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
        if output_normalizer is not None:
            self.layers.append(output_normalizer)
        
        # Set the network
        self.network = torch.nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.network(x)

    def get_latent_data(self, x):
        pass
    
    def fit(self, x_train, y_train, **kwargs):

        # Pass any optimizer parameters to the individual optimizer
        for key in kwargs:
            training_parameters[key] = kwargs[key]
        
        # Set the loss
        loss_function = get_loss(training_parameters)

        # Set the optimizer
        optimizer = get_optimizer(training_parameters, self.network.params())

        # Get the scheduler
        scheduler = get_lr_scheduler(training_parameters, optimizer)

        # Get the dataloader
        dl = get_dataloader(training_parameters, x_train, y_train)

        # save the training loss
        training_loss = []

        # For all the epochs
        for epoch in range(training_parameters['epochs']):
            
            # The epoch loss
            epoch_loss = 0

            # For all the batches
            for x_batch, y_batch in tqdm(dl):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                y_pred = self.forward(x_batch)
                loss = loss_function(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                # Add to the epoch loss
                epoch_loss += loss.item()
            
            # Append to the training loss list
            training_loss.append(epoch_loss)
            
            # Step the scheduler
            if scheduler is not None:
                scheduler.step()
            
        return training_loss
        


        


        





