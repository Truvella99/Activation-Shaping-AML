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

#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.activations = []

    def hook1(self, module, input, output):
        # print('\nForward hook running...')
        self.activations.append(output.clone().detach())

    def hook2(self, module, input, output):
        # Apply the activation shaping function to the source data
        # print('Backward hook running...')
        output_A = torch.where(output <= 0, 0.0, 1.0)
        M = self.activations.pop(0)
        M = torch.where(M <= 0, 0.0, 1.0)
        result = output_A * M
        return result

    def forward(self, src_x, targ_x=None):
        print('\nForward...')

        hooks = []
        hooks2 = []

        if targ_x is not None:  # Sono in train
            for layer in self.resnet.modules():
                if isinstance(layer, nn.Conv2d):
                    hooks.append(layer.register_forward_hook(self.hook1))
            self.resnet(targ_x)
            for h in hooks:
                h.remove()

            for layer in self.resnet.modules():
                if isinstance(layer, nn.Conv2d):
                    hooks2.append(layer.register_forward_hook(self.hook2))

            src_logits = self.resnet(src_x)

            for h in hooks2:
                h.remove()

            return src_logits
        else:
            return self.resnet(src_x)    

        """
        # Aggiungi hook per ottenere target_activation_maps
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.activation_shaping_hook)
                self.resnet(targ_x)
          
        hooks2.append(layer.register_forward_hook(self.activation_shaping_hook2))layer(src_x)"""

# Now, you can train your model using your source domain data and apply it to the target domain
# Ensure to experiment with different configurations of inserting the activation shaping layer