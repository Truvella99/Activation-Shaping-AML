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

###############################################
#                  PARAMETERS
###############################################
RATIO_OF_ONES = 0.6
K = 0.5 # between 0 and 1 in percentage (ex. 0.2 => 20%)
# Conv2d Layers of the Network:
# Set the values to True to attach the hooks on the corresponding layer
LAYERS = {
    'conv1': False,
    'layer1.0.conv1': False,
    'layer1.0.conv2': False,
    'layer1.1.conv1': False,
    'layer1.1.conv2': False,
    'layer2.0.conv1': False,
    'layer2.0.conv2': False,
    'layer2.0.downsample.0': False,
    'layer2.1.conv1': False,
    'layer2.1.conv2': False,
    'layer3.0.conv1': False,
    'layer3.0.conv2': False,
    'layer3.0.downsample.0': False,
    'layer3.1.conv1': False,
    'layer3.1.conv2': False,
    'layer4.0.conv1': False,
    'layer4.0.conv2': False,
    'layer4.0.downsample.0': False,
    'layer4.1.conv1': False,
    'layer4.1.conv2': False
}

# FUNCTION TO DECIDE WHERE ATTACH THE HOOK
# It returns the boolean value that corresponds to the name of the layer passed
def attach_hook(name=None):
    for layer_name, attach in LAYERS.items():
        if layer_name == name:
            return attach

###############################################
#                  POINT 1/3
###############################################


class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        self.activations = []

    def attach_get_activation_maps_hooks(self):
        for name,layer in self.resnet.named_modules():
            if isinstance(layer, nn.Conv2d) and attach_hook(name):
                self.hooks.append(layer.register_forward_hook(self.get_activation_maps_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()

    def attach_apply_activation_maps_hooks(self):
        for name,layer in self.resnet.named_modules():
            if isinstance(layer, nn.Conv2d) and attach_hook(name):
                self.hooks.append(layer.register_forward_hook(self.apply_activation_maps_hook))

    def get_activation_maps_hook(self, module, input, output):
        self.activations.append(output.clone().detach())

    def apply_activation_maps_hook(self, module, input, output):
        # Apply the activation shaping function to the source data
        output_A = torch.where(output <= 0, 0.0, 1.0)
        M = self.activations.pop(0)
        M = torch.where(M <= 0, 0.0, 1.0)
        result = output_A * M
        return result
    
    def forward(self, x):
        return self.resnet(x)    


###############################################
#                  POINT 2
###############################################


class RAMResNet18(nn.Module):
    def __init__(self):
        super(RAMResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []

    def attach_random_activation_maps_hooks(self):
        for name,layer in self.resnet.named_modules():
            if isinstance(layer, nn.Conv2d) and attach_hook(name):
                self.hooks.append(layer.register_forward_hook(self.random_activation_maps_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()
    
    def random_activation_maps_hook(self, module, input, output):
        # Apply the activation shaping function to the source data
        output_A = torch.where(output <= 0, 0.0, 1.0)
        # Specify the desired ratio of 1s (e.g., 0.3 for 30% of 1s)
        # YOU SHOULD TRY DIFFERENT RATIOS ==> CHANGE desired_ratio_of_ones FOR EXPERIMENTING
        desired_ratio_of_ones = RATIO_OF_ONES  # STARTING FROM M MADE OF ONLY ONES
        # Calculate the number of elements to be set to 1 based on the desired ratio
        num_ones = int(desired_ratio_of_ones * output_A.numel())
        # Create a tensor with the same shape filled with 0s
        zeros_and_ones_tensor = torch.zeros_like(output_A, dtype=torch.float32)
        # Flatten the tensor to 1D for indexing
        flat_zeros_and_ones_tensor = zeros_and_ones_tensor.view(-1)
        # Set a random subset of elements to 1 based on the desired ratio
        indices_to_set_to_one = torch.randperm(flat_zeros_and_ones_tensor.numel())[:num_ones]
        flat_zeros_and_ones_tensor[indices_to_set_to_one] = 1.0
        # Reshape the tensor back to its original shape
        M = flat_zeros_and_ones_tensor.view(output_A.shape)
        result = output_A * M
        return result
    
    def forward(self, x):
        return self.resnet(x)     

###############################################
#                  EXTENSION 2
###############################################
        
class EXTASHResNet18(nn.Module):
    def __init__(self,variation = None):
        super(EXTASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        self.activations = []
        self.variation = variation

    def attach_get_activation_maps_hooks(self):
        for name,layer in self.resnet.named_modules():
            if isinstance(layer, nn.Conv2d) and attach_hook(name):
                self.hooks.append(layer.register_forward_hook(self.get_activation_maps_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()

    def attach_apply_activation_maps_hooks(self):
        for name,layer in self.resnet.named_modules():
            if isinstance(layer, nn.Conv2d) and attach_hook(name):
                self.hooks.append(layer.register_forward_hook(self.apply_activation_maps_hook))

    def get_activation_maps_hook(self, module, input, output):
        self.activations.append(output.clone().detach())

    def apply_activation_maps_hook(self, module, input, output):
        # Apply the activation shaping function to the source data
        output_A = output
        M = self.activations.pop(0)
        # BINARIZE AND TOP K ONLY ON VARIANT 2
        if self.variation==2:
            M = torch.where(M <= 0, 0.0, 1.0)
            k = int(K * output_A.numel())
            # Step 1: Get the indices of the top k values in A
            topk_values, topk_indices = torch.topk(output_A.view(-1), k, sorted=True)
            # Step 2: Create a mask for the top k values
            mask = torch.zeros_like(output_A, dtype=torch.bool).view(-1)
            mask[topk_indices] = True
            mask = mask.view_as(output_A)
            # Step 3: Set elements not in the top k in M to 0
            M[~mask] = 0.0
        # VARIANT 1, KEEP M AS IT IS AND SIMPLY MULTIPLY BY A             
        result = output_A * M
        return result
    
    def forward(self, x):
        return self.resnet(x)


class EXTRAMResNet18(nn.Module):
    def __init__(self,variation=None):
        super(EXTRAMResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = []
        self.variation = variation

    def attach_random_activation_maps_hooks(self):
        for name,layer in self.resnet.named_modules():
            if isinstance(layer, nn.Conv2d) and attach_hook(name):
                self.hooks.append(layer.register_forward_hook(self.random_activation_maps_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()
    
    def random_activation_maps_hook(self, module, input, output):
        # Apply the activation shaping function to the source data
        output_A = output
        # Specify the desired ratio of 1s (e.g., 0.3 for 30% of 1s)
        # YOU SHOULD TRY DIFFERENT RATIOS ==> CHANGE desired_ratio_of_ones FOR EXPERIMENTING
        desired_ratio_of_ones = RATIO_OF_ONES  # STARTING FROM M MADE OF ONLY ONES
        # Calculate the number of elements to be set to 1 based on the desired ratio
        num_ones = int(desired_ratio_of_ones * output_A.numel())
        # Create a tensor with the same shape filled with 0s
        zeros_and_ones_tensor = torch.zeros_like(output_A, dtype=torch.float32)
        # Flatten the tensor to 1D for indexing
        flat_zeros_and_ones_tensor = zeros_and_ones_tensor.view(-1)
        # Set a random subset of elements to 1 based on the desired ratio
        indices_to_set_to_one = torch.randperm(flat_zeros_and_ones_tensor.numel())[:num_ones]
        flat_zeros_and_ones_tensor[indices_to_set_to_one] = 1.0
        # Reshape the tensor back to its original shape
        M = flat_zeros_and_ones_tensor.view(output_A.shape)
        # BINARIZE AND TOP K ONLY ON VARIANT 2
        if self.variation==2:
            M = torch.where(M <= 0, 0.0, 1.0)
            k = int(K * output_A.numel())
            # Step 1: Get the indices of the top k values in A
            topk_values, topk_indices = torch.topk(output_A.view(-1), k, sorted=True)
            # Step 2: Create a mask for the top k values
            mask = torch.zeros_like(output_A, dtype=torch.bool).view(-1)
            mask[topk_indices] = True
            mask = mask.view_as(output_A)
            # Step 3: Set elements not in the top k in M to 0
            M[~mask] = 0.0
        # VARIANT 1 KEEP M AS IT IS AND SIMPLY MULTIPLY IT WITH A 
        result = output_A * M
        return result
    
    def forward(self, x):
        return self.resnet(x)