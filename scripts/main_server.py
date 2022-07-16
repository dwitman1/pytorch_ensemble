import torch
from torchensemble.normalizers import TorchStandardScaler, TorchInverseStandardScaler
from torchensemble.architectures import FFNet
from torchensemble.ensemble import EnsembleRegressor
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class Train(BaseModel):
    # Descriptive name of the model
    name: str 
    # number of features/outputs
    n_features: int
    n_outputs: int
    # Information on the network architecture
    # n_outputs x n_networks x n_hidden_layers
    hidden_sizes: List[List[List[int]]]
    # Whether to do input/output Normalization
    feature_normalizer: Union[str, None] = 'TorchStandardScaler'
    # TODO need to add this functionality
    output_normalizer: Union[str, List[str], None] = 'TorchInverseStandardScaler'
    # Data to be used for training/validation
    x_train: List[List[float]]
    y_train: List[List[float]]
    x_val: List[List[float]]
    y_val: List[List[float]]

@app.post("/train")
async def train(config: Train):

    # Extract the training data
    x_train = torch.FloatTensor(config.x_train)
    y_train = torch.FloatTensor(config.y_train)
    # And the validation data
    x_val = torch.FloatTensor(config.x_val)
    y_val = torch.FloatTensor(config.y_val)

    # Build the feature normalizer
    if config.feature_normalizer is None:
        feature_normalizer = None
    elif config.feature_normalizer == 'TorchStandardScaler':
        feature_normalizer = TorchStandardScaler().fit(x_train)
    else:
        return {'Error': 'Unrecognized feature normalizer'}

    # Build the output normalizer
    if config.output_normalizer is None:
        output_normalizer = [None for _ in range(config.n_outputs)]
    elif config.output_normalizer == 'TorchInverseStandardScaler':
        output_normalizer = [TorchInverseStandardScaler().fit(y_train[:,[i_output]]) for i_output in range(config.n_outputs)]
    else:
        return {'Error': 'Unrecognized output normalizer'}
    
    # Create the networks
    networks = []
    for i_output in range(config.n_outputs):
        networks_output = []
        for hidden_sizes in config.hidden_sizes[i_output]:
            networks_output.append(FFNet(config.n_features, hidden_sizes, 1, feature_normalizer=feature_normalizer,
                                         output_normalizer=output_normalizer[i_output]))
        networks.append(networks_output)
    
    # And create the Ensemble
    ensemble = EnsembleRegressor(config.n_features, networks, config.n_outputs)

    # Fit it
    ensemble.fit(x_train, y_train, val_data=(x_val, y_val))

    # print the val error
    y_pred_val = ensemble(x_val).detach()
    print(ensemble.error(y_pred_val, y_val))

    return {'Status': 'Success'}


