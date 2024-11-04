import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
import math



def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes):
    device = alpha.get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha):
    device = y.get_device()
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):

    loglikelihood = loglikelihood_loss(y, alpha)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return A + kl_div



def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step: int = 10):
    alpha = exp_evidence(output)
    loss = torch.mean(
        edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss


class EvidentialLoss(nn.Module):
    def __init__(
        self, logvar_eps=1e-4, resi_min=1e-4, resi_max=1e3, num_classes: int = 3
    ) -> None:
        super(EvidentialLoss, self).__init__()
        self.logvar_eps = logvar_eps
        self.resi_min = resi_min
        self.resi_max = resi_max
        self.loss_fnc = edl_digamma_loss
        self.num_classes = num_classes

    def _adapt_shape(self, tensor, num_classes: int = 3):
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = torch.reshape(tensor, (-1, num_classes))
        return tensor

    def forward(self, mean: Tensor, target: Tensor, epoch: int):
        mean = self._adapt_shape(mean, num_classes=self.num_classes)
        target = self._adapt_shape(target, num_classes=self.num_classes)

        l = self.loss_fnc(mean, target, epoch, self.num_classes)
        # print("bbbbbbbbbbbbbbbbbbbbbbb")
        # print(l)
        return l



class RecLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        alpha_eps=1e-4,
        beta_eps=1e-4,
        resi_min=1e-4,
        resi_max=1e3,
        num_classes: int = 2,
    ) -> None:
        super(RecLoss, self).__init__()
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max
        self.rec_loss_fnc = EvidentialLoss(num_classes=num_classes)

    def forward(
        self,
        mean: Tensor,  ## before sm
        mask: Tensor,  ## mask
        epoch: int = 0,
    ):
        ## mean and yhat is before softmax

        ##target1 is the base model output for identity mapping
        ##target2 is the ground truth for the GenGauss loss

        ## need to convert from onehot [B, 3, W, H] -> [B, 1, W, H]
        alpha = torch.exp(mean)
        mean = mean.softmax(dim=1)

        l = self.rec_loss_fnc(mean=mean, target=mask, epoch=epoch)
        # print("ccccccccccccccccccccccc")
        # print(l)
        # sum = alpha.sum(dim=1)
        # alpha_tilde = alpha/sum.unsqueeze(1)
        # yvar = ((alpha_tilde*(1-alpha_tilde))/(sum+1).unsqueeze(1))
        # resi = mean - mask
        # print(resi.shape)
        # print(yvar.shape)
        # cov = (resi - resi.mean(dim=(-2,-1), keepdims = True))*(yvar - yvar.mean(dim=(-2,-1), keepdims = True))
        # pearson_corr = cov.mean()/ (resi.std()*yvar.std())
        # l = l2
        # print(l1)
        # print(l2)
        # print(pearson_corr)
        return l