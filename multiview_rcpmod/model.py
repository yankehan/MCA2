import torch
import torch.nn as nn
import torch.nn.functional as F

class ViewEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=128,
                 activation='relu', batchnorm=True):
        super(ViewEncoder, self).__init__()
        self._activation = activation
        self._batchnorm = batchnorm
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            if i < len(encoder_dims) - 2:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError(f'Unknown activation type {self._activation}')
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)

class ViewDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims=[256, 512], output_dim=768,
                 activation='relu', batchnorm=True):
        super(ViewDecoder, self).__init__()
        self._activation = activation
        self._batchnorm = batchnorm
        decoder_dims = [latent_dim] + hidden_dims + [output_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:
                if self._batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError(f'Unknown activation type {self._activation}')
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z):
        return self.decoder(z)

class MultiViewContrastiveModel(nn.Module):
    def __init__(self, view_dims, latent_dim=128, hidden_dims=[512, 256],
                 activation='relu', batchnorm=True):
        super(MultiViewContrastiveModel, self).__init__()
        self.view_names = list(view_dims.keys())
        self.latent_dim = latent_dim
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for view_name, input_dim in view_dims.items():
            self.encoders[view_name] = ViewEncoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                activation=activation,
                batchnorm=batchnorm
            )
            decoder_hidden_dims = list(reversed(hidden_dims))
            self.decoders[view_name] = ViewDecoder(
                latent_dim=latent_dim,
                hidden_dims=decoder_hidden_dims,
                output_dim=input_dim,
                activation=activation,
                batchnorm=batchnorm
            )

    def encode(self, views_dict):
        z_dict = {}
        for view_name, x in views_dict.items():
            if view_name in self.encoders:
                z_dict[view_name] = self.encoders[view_name](x)
        return z_dict

    def decode(self, z_dict):
        recon_dict = {}
        for view_name, z in z_dict.items():
            if view_name in self.decoders:
                recon_dict[view_name] = self.decoders[view_name](z)
        return recon_dict

    def forward(self, views_dict):
        z_dict = self.encode(views_dict)
        recon_dict = self.decode(z_dict)
        return z_dict, recon_dict

    def get_fused_representation(self, views_dict, method='mean'):
        z_dict = self.encode(views_dict)
        z_list = [z_dict[view_name] for view_name in self.view_names if view_name in z_dict]
        if method == 'mean':
            z_fused = torch.stack(z_list).mean(dim=0)
        elif method == 'max':
            z_fused = torch.stack(z_list).max(dim=0)[0]
        elif method == 'concat':
            z_fused = torch.cat(z_list, dim=1)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        return z_fused

    def to_device(self, device):
        self.to(device)
        return self

def normalize_embeddings(embeddings, method='minmax'):
    if method == 'minmax':
        min_val = embeddings.min(dim=0, keepdim=True)[0]
        max_val = embeddings.max(dim=0, keepdim=True)[0]
        normalized = (embeddings - min_val) / (max_val - min_val + 1e-8)
        stats = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    elif method == 'standard':
        mean = embeddings.mean(dim=0, keepdim=True)
        std = embeddings.std(dim=0, keepdim=True)
        normalized = (embeddings - mean) / (std + 1e-8)
        stats = {'mean': mean, 'std': std, 'method': 'standard'}
    elif method == 'l2':
        normalized = F.normalize(embeddings, p=2, dim=1)
        stats = {'method': 'l2'}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return normalized, stats

def denormalize_embeddings(normalized, stats):
    method = stats['method']
    if method == 'minmax':
        embeddings = normalized * (stats['max'] - stats['min']) + stats['min']
    elif method == 'standard':
        embeddings = normalized * stats['std'] + stats['mean']
    elif method == 'l2':
        embeddings = normalized
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return embeddings
