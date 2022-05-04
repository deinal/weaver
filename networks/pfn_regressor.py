import torch
import torch.nn as nn


class ParticleFlowNetwork(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, input_dims, Phi_sizes, use_bn=False, **kwargs):
        super(ParticleFlowNetwork, self).__init__(**kwargs)
        # input bn
        self.input_bn = nn.BatchNorm1d(input_dims) if use_bn else nn.Identity(),
        # per-particle functions
        phi_layers = []
        for i in range(len(Phi_sizes)):
            phi_layers.append(nn.Sequential(
                nn.Conv1d(input_dims if i == 0 else Phi_sizes[i - 1], Phi_sizes[i], kernel_size=1),
                nn.BatchNorm1d(Phi_sizes[i]) if use_bn else nn.Identity(),
                nn.ReLU())
            )
        self.phi = nn.Sequential(*phi_layers)

    def forward(self, x, mask=None):
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        x = self.phi(x)
        if mask is not None:
            x *= mask.bool().float()
        return x.sum(-1)


class ParticleFlowNetworkRegressor(nn.Module):
    def __init__(self, 
        ch_input_dims, 
        ne_input_dims, 
        sv_input_dims,
        jet_input_dims,
        num_classes,
        Phi_sizes=[100, 100, 100],
        F_sizes=[(100, 0.0), (100, 0.0), (100, 0.0)],
        use_bn=False,
        **kwargs
    ):
        super(ParticleFlowNetworkRegressor, self).__init__(**kwargs)

        self.ch_pfn = ParticleFlowNetwork(ch_input_dims, Phi_sizes, use_bn)
        self.ne_pfn = ParticleFlowNetwork(ne_input_dims, Phi_sizes, use_bn)
        self.sv_pfn = ParticleFlowNetwork(sv_input_dims, Phi_sizes, use_bn)

        # global functions
        f_layers = []
        for i in range(len(F_sizes)):
            out_units, drop_rate = F_sizes[i]
            f_layers.append(nn.Sequential(
                nn.Linear(jet_input_dims + 3 * Phi_sizes[-1] if i == 0 else F_sizes[i - 1][0], out_units),
                nn.ReLU(),
                nn.Dropout(drop_rate))
            )
        f_layers.append(nn.Linear(F_sizes[-1][0], num_classes))
        self.fc = nn.Sequential(*f_layers)

    def forward(self, ch_features, ch_mask, ne_features, ne_mask, sv_features, sv_mask, jet_features):
        ch_x = self.ch_pfn(ch_features, ch_mask)
        ne_x = self.ne_pfn(ne_features, ne_mask)
        sv_x = self.sv_pfn(sv_features, sv_mask)
        jet_x = jet_features
        x = torch.cat((ch_x, ne_x, sv_x, jet_x), dim=1)
        output = self.fc(x)
        return output


def get_model(data_config, **kwargs):
    dropout = kwargs.get('dropout', 0.0)
    num_conv_layers = kwargs.get('num_conv_layers', 3)
    conv_dim = kwargs.get('conv_dim', 100)
    num_fc_layers = kwargs.get('num_fc_layers', 3)
    fc_dim = kwargs.get('fc_dim', 100)

    Phi_sizes = num_conv_layers * [conv_dim]
    F_sizes = num_fc_layers * [(fc_dim, dropout)]

    ch_input_dims = len(data_config.input_dicts['ch_features'])
    ne_input_dims = len(data_config.input_dicts['ne_features'])
    sv_input_dims = len(data_config.input_dicts['sv_features'])
    jet_input_dims = len(data_config.input_dicts['jet_features'])
    num_classes = 1
    model = ParticleFlowNetworkRegressor(
        ch_input_dims, ne_input_dims, sv_input_dims, jet_input_dims, num_classes, 
        Phi_sizes=Phi_sizes, F_sizes=F_sizes, use_bn=kwargs.get('use_bn', False)
    )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'output_shapes': {'output': (1, 1)},
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
    }
    model_info['dynamic_axes']['jet_features'] = {0: 'N', 1: 'n_jet'}
    model_info['dynamic_axes']['output'] = {0: 'N'}

    # print(model, model_info)
    return model, model_info


class CustomL1Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomL1Loss, self).__init__()

    def forward(self, y_pred, y_true):
        mask = torch.logical_and(y_true > -1, y_true < 1)
        return torch.mean(torch.abs(y_pred - y_true) * mask)


def get_loss(data_config, **kwargs):
    return CustomL1Loss()
