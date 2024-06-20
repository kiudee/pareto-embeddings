import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def dominance_loss(output, target, big_value=100.0):
    """ Loss ensuring that each object which is not chosen, will be dominated. """
    n_obj = output.shape[-2]
    device = output.device
    return (
        (
            (
                torch.sum(F.relu(1 + output[..., None, :] - output[:, None]), dim=-1)
                + big_value * torch.eye(n_obj, device=device)[None, :, :]
            )
            * (~target[..., :, None]).to(device)
        )
        .min(axis=-1)[0]
        .sum(axis=-1)
    )


def dominance_loss_soft(output, target, big_value=100.0):
    """ Loss ensuring that each object which is not chosen, will be dominated. """
    n_obj = output.shape[-2]
    device = output.device
    return (
        (
            (
                torch.sum(
                    F.softplus(1 + output[..., None, :] - output[:, None]), dim=-1
                )
                + big_value * torch.eye(n_obj, device=device)[None, :, :]
            )
            * (~target[..., :, None]).to(device)
        )
        .min(axis=-1)[0]
        .sum(axis=-1)
    )


def anti_dominance_loss(output, target, big_value=100.0):
    """Loss used to ensure that the chosen objects are not dominated. """
    n_obj = output.shape[-2]
    device = output.device
    return torch.sum(
        F.relu(
            ((1 + output[..., None, :] - output[:, None])).min(-1)[0]
            * target[..., None, :].to(device)
            - torch.eye(n_obj, device=device) * big_value
        ),
        axis=(-1, -2),
    )


def anti_dominance_loss_soft(output, target, big_value=100.0):
    """Loss used to ensure that the chosen objects are not dominated. """
    n_obj = output.shape[-2]
    device = output.device
    return torch.sum(
        F.softplus(
            ((1 + output[..., None, :] - output[:, None])).min(-1)[0]
            * target[..., None, :].to(device)
            - torch.eye(n_obj, device=device) * big_value
        ),
        axis=(-1, -2),
    )


def l2_loss(output, target):
    return torch.sum(torch.norm(output, dim=-1), dim=-1)


# def min_distance_loss(output, target):
#     dist = F.relu(1 - torch.cdist(output, output))
#     dist -= torch.eye(dist.shape[-1]).to(device)
#     return torch.sum(torch.sum(dist * target[..., :, None], dim=-1), dim=-1)


def inter_distance_loss(inp, output):
    device = output.device
    in_dist = torch.cdist(inp, inp).to(device)
    out_dist = torch.cdist(output, output).to(device)
    return torch.norm(in_dist - out_dist, p=2, dim=(-1, -2))


class ParetoEmbeddingLoss(_Loss):
    def __init__(
        self,
        zero_weight=0.0,
        dominance_weight=1.0,
        anti_dominance_weight=1.0,
        inter_weight=0.01,
        big_constant=100.0,
        **kwargs
    ):
        super(ParetoEmbeddingLoss, self).__init__(**kwargs)
        self.zero_weight = zero_weight
        self.dominance_weight = dominance_weight
        self.anti_dominance_weight = anti_dominance_weight
        self.inter_weight = inter_weight
        self.big_constant = big_constant

    def forward(self, y_pred, y_true, X, **kwargs):
        total_loss = self.zero_weight * l2_loss(y_pred, y_true)
        total_loss += self.dominance_weight * dominance_loss(
            y_pred, y_true, big_value=self.big_constant
        )
        total_loss += self.anti_dominance_weight * anti_dominance_loss(
            y_pred, y_true, big_value=self.big_constant
        )
        total_loss += self.inter_weight * inter_distance_loss(X, y_pred)

        return torch.mean(total_loss)


class ParetoEmbeddingLossSoft(_Loss):
    def __init__(
        self,
        zero_weight=0.0,
        dominance_weight=1.0,
        anti_dominance_weight=1.0,
        inter_weight=0.01,
        big_constant=100.0,
        **kwargs
    ):
        super(ParetoEmbeddingLossSoft, self).__init__(**kwargs)
        self.zero_weight = zero_weight
        self.dominance_weight = dominance_weight
        self.anti_dominance_weight = anti_dominance_weight
        self.inter_weight = inter_weight
        self.big_constant = big_constant

    def forward(self, y_pred, y_true, X, **kwargs):
        total_loss = self.zero_weight * l2_loss(y_pred, y_true)
        total_loss += self.dominance_weight * dominance_loss_soft(
            y_pred, y_true, big_value=self.big_constant
        )
        total_loss += self.anti_dominance_weight * anti_dominance_loss_soft(
            y_pred, y_true, big_value=self.big_constant
        )
        total_loss += self.inter_weight * inter_distance_loss(X, y_pred)

        return torch.mean(total_loss)
