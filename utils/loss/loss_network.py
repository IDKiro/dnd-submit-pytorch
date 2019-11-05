from collections import namedtuple

import torch
import torchvision.models.vgg as vgg

LossOutput = namedtuple(
    "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])


class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

class PerceptualLoss(torch.nn.Module):
    def __init__(self, content_weight=0.01):
        super(PerceptualLoss, self).__init__()
        self.weight = content_weight
        self.mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            self.loss_network = LossNetwork()

    def forward(self, pred, truth):
        features_pred = self.loss_network(pred)
        features_truth = self.loss_network(truth)

        loss = self.mse_loss(pred, truth) + \
            self.weight * self.mse_loss(features_pred[2], features_truth[2])

        return loss
