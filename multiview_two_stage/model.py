import torch
import torch.nn as nn
from sklearn.decomposition import PCA

class ViewImportanceGate(nn.Module):
    def __init__(
        self,
        view_dims,
        num_views,
        hidden_dims=None,
        temperature=1.0,
        proj_dim=128,
        debug_print=True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []
        self.temperature = float(temperature)
        self.view_names = list(view_dims.keys())
        self.num_views = int(num_views)
        self.proj_dim = proj_dim
        self.debug_print = bool(debug_print)
        self.view_pcas = {}
        self.pca_fitted = {name: False for name in self.view_names}
        for name in self.view_names:
            n_components = min(proj_dim, view_dims[name])
            self.view_pcas[name] = PCA(n_components=n_components)
        dims = [proj_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Sigmoid())
        self.logit_mlp = nn.Sequential(*layers)

    def forward(self, views_dict):
        proj_list = []
        for name in self.view_names:
            x = views_dict[name]
            x_np = x.detach().cpu().numpy()
            if not self.pca_fitted[name]:
                self.view_pcas[name].fit(x_np)
                self.pca_fitted[name] = True
            x_pca = self.view_pcas[name].transform(x_np)
            x_tensor = torch.tensor(x_pca, dtype=x.dtype, device=x.device)
            if x_tensor.shape[1] < self.proj_dim:
                padding = torch.zeros(x_tensor.shape[0], self.proj_dim - x_tensor.shape[1],
                                    dtype=x.dtype, device=x.device)
                x_tensor = torch.cat([x_tensor, padding], dim=1)
            proj_list.append(x_tensor)
        logits_list = []
        for x in proj_list:
            logit = self.logit_mlp(x)
            logits_list.append(logit)
        logits = torch.cat(logits_list, dim=1)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        weight = torch.sigmoid(logits)
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-12)
        if self.debug_print:
            print("Gate output weight[0]=", weight[0])
            self.debug_print = False
        return weight

    def reset_to_uniform(self):
        eps = 1e-3
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=eps)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
                 activation='relu', batchnorm=True, use_view_gate=True, view_gate_hidden_dims=None,
                 view_gate_temperature=1.0):
        super(MultiViewContrastiveModel, self).__init__()
        self.view_names = list(view_dims.keys())
        self.latent_dim = latent_dim
        self.use_view_gate = use_view_gate
        self.view_gate_mode = 'learned' if self.use_view_gate else 'off'
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        print("Gate hidden dims:", view_gate_hidden_dims)
        self.view_gate = ViewImportanceGate(
            num_views=len(self.view_names),
            view_dims=view_dims,
            hidden_dims=view_gate_hidden_dims,
            temperature=view_gate_temperature,
            proj_dim=128
        )
        if self.use_view_gate:
            print("Gate ON: will use gate weights")
        else:
            print("Gate OFF: gate created but weights not used")
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
    def compute_view_weights(self, views_dict):
        if not self.use_view_gate:
            return None
        used_view_names = [v for v in self.view_names if v in views_dict]
        used_views_dict = {v: views_dict[v] for v in used_view_names}
        if getattr(self, 'view_gate_mode', 'learned') == 'uniform':
            batch_size = list(used_views_dict.values())[0].shape[0]
            num_used_views = len(used_view_names)
            first_view = list(used_views_dict.values())[0]
            weights = torch.full(
                (batch_size, num_used_views),
                1.0 / max(1, num_used_views),
                dtype=first_view.dtype,
                device=first_view.device,
            )
        else:
            weights = self.view_gate(used_views_dict)
        weights_dict = {view_name: weights[:, i] for i, view_name in enumerate(used_view_names)}
        return weights, weights_dict
    def set_view_gate_mode(self, mode: str):
        self.view_gate_mode = str(mode)
