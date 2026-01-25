import torch
import torch.nn as nn
import torch.nn.functional as F
class Autoencoder(nn.Module):
    def __init__(self, data_dim, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.rep_dim = latent_dim
        self.data_dim = data_dim
        if data_dim <= 100:
            en_layers_num = [self.data_dim, 128, self.rep_dim]
        elif data_dim <= 1000:
            en_layers_num = [self.data_dim, 256, 128, self.rep_dim]
        else:
            en_layers_num = [self.data_dim, 512, 256, self.rep_dim]
        self.encoder = self.encode(en_layers_num)
        de_layers_num = list(reversed(en_layers_num))
        self.decoder = self.decode(de_layers_num)
    def encode(self, layers_num):
        if len(layers_num) > 2:
            encoded_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Tanh())
            for i in range(1, len(layers_num) - 2):
                encoded_output = nn.Sequential(encoded_output, nn.Linear(layers_num[i], layers_num[i+1]), nn.Tanh())
            encoded_output = nn.Sequential(encoded_output, nn.Linear(layers_num[len(layers_num)-2],
                                                                     layers_num[len(layers_num)-1]))
        else:
            encoded_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Tanh())
        return encoded_output
    def decode(self, layers_num):
        if len(layers_num) > 2:
            decode_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Tanh())
            for i in range(1, len(layers_num) - 2):
                decode_output = nn.Sequential(decode_output, nn.Linear(layers_num[i], layers_num[i+1]), nn.Tanh())
            decode_output = nn.Sequential(decode_output, nn.Linear(layers_num[len(layers_num)-2],
                                                                   layers_num[len(layers_num)-1]), nn.Sigmoid())
        else:
            decode_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Sigmoid())
        return decode_output
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
class NCMODModel:
    def __init__(self, view_dims, latent_dim=32, device='cpu'):
        self.view_names = list(view_dims.keys())
        self.num_views = len(self.view_names)
        self.latent_dim = latent_dim
        self.device = device
        self.autoencoders = {}
        for view_name, dim in view_dims.items():
            self.autoencoders[view_name] = Autoencoder(dim, latent_dim).to(device)
    def get_view_net(self, view_name):
        return self.autoencoders[view_name]
    def to(self, device):
        self.device = device
        for view_name in self.view_names:
            self.autoencoders[view_name] = self.autoencoders[view_name].to(device)
        return self
