import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(input):
    """Given a set V of m vectors, function returns the matrix of all possible inner products of V

    Args:
        input (tensor): Input is of shape B x C x W x H

    Returns:
        tensor: Normalized gram matrix of input tensor
    """
    b, d, w, h = input.size()
    tensor = input.view(b * d, w * h)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(b * d * w * h)


def total_variation_loss(input):
    """Computes total variation noise of an input tensor

    Args:
        input (tensor): Input is of shape B x C x W x H


    Returns:
        Total variation loss of the input tensor
    """
    b, d, w, h = input.size()
    x = torch.pow(input[:, :, 1:, :] - input[:, :, :-1, :], 2).sum()
    y = torch.pow(input[:, :, :, 1:] - input[:, :, :, :-1], 2).sum()
    loss = (x + y) / (b * d * w * h)
    return loss


class ContentLoss(nn.Module):
    """
        Computes the MSE Loss between input image and the image to be generated
    """

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """
        Calculates the mean-squared distance between the Gram matrix of input image and
        the Gram matrix of the image to be generated (target image)
    """

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        gram_input = gram_matrix(input)
        self.loss = F.mse_loss(gram_input, self.target)
        return input


class Normalization(nn.Module):
    """
        Normalizes input images with pre-defined mean and standard deviation
    """

    def __init__(self, mean, sd):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.sd = torch.tensor(sd).view(-1, 1, 1)

    def forward(self, input):
        return (input - self.mean) / self.sd

