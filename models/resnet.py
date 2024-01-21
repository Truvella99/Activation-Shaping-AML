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
        # print('Forward hook running...')
        self.activations.append(output.clone().detach())

    def hook2(self, module, input, output):
        # Apply the activation shaping function to the source data
        # print('Backward hook running...')
        output_A = torch.where(output <= 0, 0.0, 1.0)
        M = self.activations.pop(0)
        M = torch.where(M <= 0, 0.0, 1.0)
        result = output_A * M
        return result

    def attach_hook(self,mode,counter,step=None):
      # TRY MULTIPLE CONFIGURATIONS
      CONV_LAYERS = 19
      FIRST_LAYER = 0
      LAST_LAYER = CONV_LAYERS
      MIDDLE_LAYER = int(CONV_LAYERS/2)
      
      if mode == 'counter_step':
          return counter % step == 0
      elif mode == 'first':
          return counter == FIRST_LAYER
      elif mode == 'middle':
          return counter == MIDDLE_LAYER
      elif mode == 'last':
          return counter == LAST_LAYER
      elif mode == 'first_middle':
          return counter == FIRST_LAYER or counter == MIDDLE_LAYER
      elif mode == 'middle_last':
          return counter == MIDDLE_LAYER or counter == LAST_LAYER
      elif mode == 'first_last':
          return counter == FIRST_LAYER or counter == LAST_LAYER
      else:
          return False
    
    def forward(self, src_x, targ_x=None):
        print('\nForward...')

        hooks = []
        hooks2 = []
        counter = 0
        step = 1 # 1 = ALL CONV2D LAYERS

        if targ_x is not None:  # Sono in train
            for layer in self.resnet.modules():
                if isinstance(layer, nn.Conv2d):
                    if self.attach_hook(mode='counter_step',counter=counter,step=step):
                      hooks.append(layer.register_forward_hook(self.hook1))
                    counter+=1
                
            self.resnet(targ_x)
            
            for h in hooks:
                h.remove()
            
            # RESET COUNTER FOR SECOND HOOK
            counter = 0
            
            for layer in self.resnet.modules():
                if isinstance(layer, nn.Conv2d):
                    if self.attach_hook(mode='counter_step',counter=counter,step=step):
                      hooks2.append(layer.register_forward_hook(self.hook2))
                    counter+=1

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

class RAMResNet18(nn.Module):
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