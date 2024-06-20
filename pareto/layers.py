import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from pareto.fspool import FSPool

__all__ = ["ParetoEmbedding", "FATE", "MeanEncoder", "FSEncoder", "PairwiseEncoder"]


class ParetoEmbedding(nn.Module):
    def __init__(self, n_features, n_embed=2, n_hidden_layers=2, n_hidden_units=32):
        super(ParetoEmbedding, self).__init__()
        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        for i in range(n_hidden_layers):
            if i == 0:
                self.fc.append(nn.Linear(n_features, n_hidden_units))
            else:
                self.fc.append(nn.Linear(n_hidden_units, n_hidden_units))
            self.bn.append(nn.LayerNorm(n_hidden_units))
        self.out = nn.Linear(n_hidden_units, n_embed)

    def forward(self, x):
        for i, layer in enumerate(self.fc):
            x = F.relu(layer(x))
            x = self.bn[i](x)
        x = torch.exp(self.out(x))
        return x


class FSEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        # TODO: Allow customizing the number of layers
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )
        self.pool = FSPool(dim, 20, relaxed=kwargs.get("relaxed", True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x, perm = self.pool(x, n_points)
        x = self.lin(x)
        return x, perm


class MeanEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        if n_points is None:
            x = x.mean(2)
        else:
            x = x.sum(2) / n_points.unsqueeze(1).float()
        x = self.lin(x)
        return x


class FATE(nn.Module):
    def __init__(
        self,
        set_encoder,
        n_input_features,
        n_set_features=16,
        n_scorer_layers=1,
        n_scorer_dim=16,
        n_scorer_output=2,
        set_encoder_args=None,
        **kwargs
    ):
        super().__init__()
        if set_encoder_args is None:
            set_encoder_args = dict()
        self.set_encoder = set_encoder(
            input_channels=n_input_features,
            output_channels=n_set_features,
            **set_encoder_args
        )
        full_dim = n_input_features + n_set_features
        layers = [nn.Conv1d(full_dim, n_scorer_dim, 1), nn.ReLU(inplace=True)]
        for _ in range(n_scorer_layers - 1):
            layers.extend(
                (nn.Conv1d(n_scorer_dim, n_scorer_dim, 1), nn.ReLU(inplace=True))
            )
        self.scorer = nn.Sequential(
            *layers, nn.Conv1d(n_scorer_dim, n_scorer_output, 1)
        )

        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, n_points=None):
        n_batch, n_obj, n_feat = x.shape
        new_x = x.transpose(1, 2).contiguous()
        embedding = self.set_encoder(new_x, n_points)
        if isinstance(embedding, tuple):
            embedding = embedding[0]
        emb_repeat = embedding.view(*embedding.shape, 1).repeat(1, 1, n_obj)
        new_x = torch.cat([new_x, emb_repeat], dim=1)
        scores = self.scorer(new_x)
        return scores.transpose(1, 2).contiguous()


def _generate_pairs(instances):
    """Generate object pairs from instances.

    This can be used to generate the individual comparisons in a first-order
    utility function model.

    >>> object_a = [0.5, 0.6]
    >>> object_b = [1.5, 1.6]
    >>> object_c = [2.5, 2.6]
    >>> object_d = [3.5, 3.6]
    >>> object_e = [4.5, 4.6]
    >>> object_f = [5.5, 5.6]
    >>> # instance = list of objects to rank
    >>> instance_a = [object_a, object_b, object_c]
    >>> instance_b = [object_d, object_e, object_f]
    >>> instances = [instance_a, instance_b]

    >>> _generate_pairs(torch.tensor(instances))
    tensor([[[0.5000, 0.6000, 0.5000, 0.6000],
             [0.5000, 0.6000, 1.5000, 1.6000],
             [0.5000, 0.6000, 2.5000, 2.6000],
             [1.5000, 1.6000, 0.5000, 0.6000],
             [1.5000, 1.6000, 1.5000, 1.6000],
             [1.5000, 1.6000, 2.5000, 2.6000],
             [2.5000, 2.6000, 0.5000, 0.6000],
             [2.5000, 2.6000, 1.5000, 1.6000],
             [2.5000, 2.6000, 2.5000, 2.6000]],
    <BLANKLINE>
            [[3.5000, 3.6000, 3.5000, 3.6000],
             [3.5000, 3.6000, 4.5000, 4.6000],
             [3.5000, 3.6000, 5.5000, 5.6000],
             [4.5000, 4.6000, 3.5000, 3.6000],
             [4.5000, 4.6000, 4.5000, 4.6000],
             [4.5000, 4.6000, 5.5000, 5.6000],
             [5.5000, 5.6000, 3.5000, 3.6000],
             [5.5000, 5.6000, 4.5000, 4.6000],
             [5.5000, 5.6000, 5.5000, 5.6000]]])
    """

    def repeat_individual_objects(instances, times):
        """Repeat each object once, immediately after the original.

        >>> repeat_individual_objects(torch.tensor(
        True
        """
        # add a dimension, so that each object is now enclosed in a singleton
        unsqueezed = instances.unsqueeze(2)

        # repeat every object (along the newly added dimension)
        # ([[object_a], [object_a]], [[object_b]], [[object_b]])
        repeated = unsqueezed.repeat(1, 1, times, 1)
        # collapse the added dimension again so that each object is on the same
        # level ([object_a], [object_a], [object_b]], [[object_b])
        return repeated.view(instances.size(0), -1, instances.size(2))

    def repeat_object_lists(instances, times):
        """Repeat the whole object list as a unit (the same as "first" but in a different order)."""
        return instances.repeat(1, times, 1)

    n_objects = instances.size(1)
    first = repeat_individual_objects(instances, n_objects)
    second = repeat_object_lists(instances, n_objects)

    # Glue the two together at the object level (the object's feature vectors
    # are concatenated)
    output_tensor = torch.cat((first, second), dim=2)
    return output_tensor


class PairwiseEncoder(nn.Module):
    def __init__(self, in_features, out_features, n_layers=1, n_units=8):
        super().__init__()

        self.n_layers = n_layers
        self.n_units = n_units
        self.out_features = out_features

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                lin_features = in_features * 2
            else:
                lin_features = n_units
            self.layers.append(
                nn.Sequential(
                    nn.Linear(lin_features, n_units),
                    nn.LayerNorm(n_units),
                    nn.ReLU(inplace=True),
                )
            )
        if n_layers > 0:
            self.out_layer = nn.Linear(n_units, out_features)
        else:
            self.out_layer = nn.Linear(in_features * 2, out_features)

    def forward(self, x):
        n_inst, n_obj, n_feat = x.shape
        pairs = _generate_pairs(x)
        for layer in self.layers:
            pairs = layer(pairs)

        out_x = self.out_layer(pairs)
        out_x = out_x.reshape(n_inst, n_obj, n_obj, self.out_features).mean(dim=-2)
        return torch.nn.functional.softplus(out_x)
