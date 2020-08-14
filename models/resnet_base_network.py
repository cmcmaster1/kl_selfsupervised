import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
import torch
from models.mlp_head import MLPHead

# class ResNet18(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(ResNet18, self).__init__()
#         if kwargs['name'] == 'resnet18':
#             resnet = models.resnet18(pretrained=False)
#         elif kwargs['name'] == 'resnet50':
#             resnet = models.resnet50(pretrained=False)

#         self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
#         self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

#     def forward(self, x):
#         h = self.encoder(x)
#         h = h.view(h.shape[0], h.shape[1])
#         return self.projetion(h)

class GreyNet50(ResNet):
    def __init__(self):
        super(GreyNet50, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)

class GreyNet18(ResNet):
    def __init__(self):
        super(GreyNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)

class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = GreyNet18()
        elif kwargs['name'] == 'resnet50':
            resnet = GreyNet50()

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)