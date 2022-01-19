import torch
from torchvision.models import resnet18, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2
import torch.nn as nn

class Resnet18(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        # self.resnet = resnet18(pretrained=pretrained, num_classes=num_classes)
        self.resnet = resnet18(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x

class Resnet50(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = resnet50(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


class Resnet101(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = resnet101(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


class Resnet152(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = resnet152(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


class Wide_Resnet50_2(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = wide_resnet50_2(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x


class Wide_Resnet101_2(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.resnet = wide_resnet101_2(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.reshape(batch_size, channels, height, width)
        x = self.resnet(x)
        x = x.reshape(batch_size, -1)

        return x