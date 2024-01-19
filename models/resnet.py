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
'''
def activation_shaping_hook(module, input, output):
    self.target_activation_maps.append(output.clone().detach())
    return output

def activation_shaping_hook2(module, input, output):
    output_A = torch.where(output <= 0, 0.0, 1.0)
    M = module.target_activation_maps.pop(0)
    M = torch.where(M <= 0, 0.0, 1.0)
    result = output_A * M
    module.product.append(result.clone().detach())
    return result
'''
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.target_activation_maps = []

    def activation_shaping_hook(self, module, input, output):
        self.target_activation_maps.append(output.clone().detach())

    def activation_shaping_hook2(self, module, input, output):
        output_A = torch.where(output <= 0, 0.0, 1.0)
        M = self.target_activation_maps.pop(0)
        M = torch.where(M <= 0, 0.0, 1.0)
        result = output_A * M
        return result

    def forward(self, src_x, targ_x):

        # Aggiungi hook per ottenere target_activation_maps
        for name, layer in self.resnet.named_children():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.activation_shaping_hook)
                self.resnet(targ_x)

        # Esegui l'input attraverso il modello
 

        print("target_activation_maps")
        # Apply the activation shaping function to the source data
        for name, layer in self.resnet.named_children():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.activation_shaping_hook2)
                """hooks2.append(layer.register_forward_hook(self.activation_shaping_hook2))
                layer(src_x)"""
                self.resnet(src_x)
        print("product")

        
        #src_x = src_x.view(src_x.shape[0] , -1)
        #src_x = self.resnet.fc(src_x)
        
        # Return the final result
        return self.resnet(src_x)



# versione funzionante ma non troppo
class ASHResNet18Bis(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.target_activation_maps = []

    def activation_shaping_hook(self, module, input, output):
        self.target_activation_maps.append(output.clone().detach())

    def activation_shaping_hook2(self, module, input, output):
        output_A = torch.where(output <= 0, 0.0, 1.0)
        M = self.target_activation_maps.pop(0)
        M = torch.where(M <= 0, 0.0, 1.0)
        result = output_A * M
        return result

    def forward(self, src_x, targ_x):
        # List to store hooks

        # Add hook to obtain target_activation_maps
        for name, layer in self.resnet.named_children():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.activation_shaping_hook)
                """hooks.append(layer.register_forward_hook(self.activation_shaping_hook))
                layer(targ_x)"""


        print("target_activation_maps")
        # Apply the activation shaping function to the source data
        for name, layer in self.resnet.named_children():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.activation_shaping_hook2)
                """hooks2.append(layer.register_forward_hook(self.activation_shaping_hook2))
                layer(src_x)"""

        print("product")

        
        #src_x = src_x.view(src_x.shape[0] , -1)
        #src_x = self.resnet.fc(src_x)
        
        # Return the final result
        return self.resnet(src_x)


"""
class CustomActivationShapingLayer(nn.Module):
    def __init__(self):
        super(CustomActivationShapingLayer, self).__init__()

    def forward(self, A, M):
        # Binarize activation map A
        A_bin = torch.where(A <= 0, torch.tensor(0.0), torch.tensor(1.0))
        
        # Binarize tensor M
        M_bin = torch.where(M <= 0, torch.tensor(0.0), torch.tensor(1.0))

        # Element-wise product of A_bin and M_bin
        output = A_bin * M_bin

        return output

class ASHResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.target_activation_maps = []
        self.product = []
        self.activation_shaping_layer = CustomActivationShapingLayer()

    def activation_shaping_function(self, output):
        self.target_activation_maps.append(output.clone().detach())
        return output

    def activation_shaping_function2(self, output, M):
        output_A = torch.where(output <= 0, 0.0, 1.0)
        M_bin = torch.where(M <= 0, torch.tensor(0.0), torch.tensor(1.0))
        result = output_A * M_bin
        self.product.append(result.clone().detach())
        return result

    def forward(self, src_x, src_y, targ_x):
        # Direct call to the activation shaping function to obtain target_activation_maps
        for name, layer in self.resnet.named_children():
            if isinstance(layer, nn.Conv2d):
                targ_x = self.activation_shaping_function(targ_x)

        print("target_activation_maps")

        # Direct call to the activation shaping function to perform the binary product
        for name, layer in self.resnet.named_children():
            if isinstance(layer, nn.Conv2d):
                src_x = self.activation_shaping_function2(src_x,self.target_activation_maps.pop(0))

        print("product")

        # Apply the custom activation shaping layer
        #src_x = self.activation_shaping_layer(src_x, targ_x)

        # Restituisce il risultato finale
        return src_x

# Create an instance of ResNet18WithActivationShaping

"""
# Now, you can train your model using your source domain data and apply it to the target domain
# Ensure to experiment with different configurations of inserting the activation shaping layer