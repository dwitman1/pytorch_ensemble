import torch

# TODO: Add max normalizer
# TODO: Add max/min normalizer
# TODO: Add combo normalizer class that takes on initialization a
#       list of normalizers and performs normalization for each column

class TorchStandardScaler(torch.nn.Module):

    def __init__(self):
        super(TorchStandardScaler, self).__init__()
        
        # Set the mean and standard deviation to zero for initialization
        self.mean = 0
        self.std = 0
    
    def fit(self, x):
        # extract the mean
        self.mean = torch.mean(x, dim=0, keepdim=True)
        # and the standard deviation
        self.std = torch.std(x, dim=0, keepdim=True)

        return self
     
    def forward(self, x):
        return (x - torch.tile(self.mean, (x.shape[0], 1)))/torch.tile(self.std, (x.shape[0], 1))
    
class TorchInverseStandardScaler(TorchStandardScaler):

    def forward(self, x):
        return x*torch.tile(self.std, (x.shape[0], 1)) + torch.tile(self.mean, (x.shape[0], 1))
