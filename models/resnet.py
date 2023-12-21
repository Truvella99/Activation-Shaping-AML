import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from globals import CONFIG

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
def activation_shaping_hook(module, input, output):
    output_A = torch.where(output <= 0, 0.0, 1.0)
    M = module.target_activation_maps.pop(0)
    M = torch.where(M <= 0, 0.0, 1.0)
    module.product.append(output_A * M)
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.target_activation_maps = []
        self.hooks = []
        self.product = []
    
    def forward(self, src_x, src_y, targ_x):
        self.target_activation_maps.append(targ_x)
        for name, layer in self.resnet.named_children():
          if isinstance(layer, nn.Conv2d):
            self.hooks.append(layer.register_forward_hook(activation_shaping_hook))
        for h in self.hooks:
          h.remove()
        return self.resnet(src_x) #torch.tensor(self.product.pop(0)).to(CONFIG.device)

######################################################
