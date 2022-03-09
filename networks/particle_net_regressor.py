import numpy as np
import torch
import torch.nn as nn

'''Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.'''


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):
    def __init__(self,
                 input_dims,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 use_fts_bn=True,
                 for_inference=False,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

    def forward(self, points, features, mask=None):
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask

        return fts.mean(dim=-1)


class FeatureConv(nn.Module):
    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ParticleNetRegressor(nn.Module):
    def __init__(
            self,
            ch_features_dims,
            ne_features_dims,
            sv_features_dims,
            jet_features_dims,
            num_classes,
            conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
            fc_params=[(128, 0.1)],
            use_fts_bn=True,
            ch_input_dropout=None,
            ne_input_dropout=None,
            sv_input_dropout=None,
            for_inference=False,
            **kwargs
        ):
        super(ParticleNetRegressor, self).__init__(**kwargs)
        self.ch_input_dropout = nn.Dropout(ch_input_dropout) if ch_input_dropout else None
        self.ne_input_dropout = nn.Dropout(ne_input_dropout) if ne_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.ch_conv = FeatureConv(ch_features_dims, 32)
        self.ne_conv = FeatureConv(ne_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        self.pn = ParticleNet(
            input_dims=32,
            conv_params=conv_params,
            use_fts_bn=use_fts_bn,
            for_inference=for_inference,
        )

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = conv_params[-1][1][-1] + jet_features_dims
            else:
                in_chn = fc_params[idx - 1][0]
            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference
        
    def forward(
            self, 
            ch_points, ch_features, ch_mask, 
            ne_points, ne_features, ne_mask, 
            sv_points, sv_features, sv_mask,
            jet_features
        ):
        
        if self.ch_input_dropout:
            ch_mask = (self.ch_input_dropout(ch_mask) != 0).float()
            ch_points *= ch_mask
            ch_features *= ch_mask
        if self.ne_input_dropout:
            ch_mask = (self.ne_input_dropout(ne_mask) != 0).float()
            ch_points *= ne_mask
            ch_features *= ne_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask

        points = torch.cat((ch_points, ne_points, sv_points), dim=2)
        features = torch.cat((
            self.ch_conv(ch_features * ch_mask) * ch_mask,
            self.ne_conv(ne_features * ne_mask) * ne_mask,
            self.sv_conv(sv_features * sv_mask) * sv_mask
        ), dim=2)
        mask = torch.cat((ch_mask, ne_mask, sv_mask), dim=2)
        x = self.pn(points, features, mask)
        jet_x = jet_features.squeeze(dim=-1)
        x = torch.cat((x, jet_x), dim=1)
        return self.fc(x)


def get_model(data_config, **kwargs):
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1), (128, 0.1)]

    ch_features_dims = len(data_config.input_dicts['ch_features'])
    ne_features_dims = len(data_config.input_dicts['ne_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    jet_features_dims = len(data_config.input_dicts['jet_features'])
    num_classes = 1
    model = ParticleNetRegressor(
        ch_features_dims, ne_features_dims, sv_features_dims, 
        jet_features_dims, num_classes,
        conv_params, fc_params,
        use_fts_bn=kwargs.get('use_fts_bn', False),
        ch_input_dropout=kwargs.get('ch_input_dropout', None),
        ne_input_dropout=kwargs.get('ne_input_dropout', None),
        sv_input_dropout=kwargs.get('sv_input_dropout', None),
        for_inference=False
    )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
        }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.SmoothL1Loss(beta=kwargs.get('loss_beta', 10))
