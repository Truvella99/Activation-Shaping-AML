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
RATIO = 0.6
K = 5
STEP = 1 # 1 = All Conv2D Layers
MODE = 'last' # Modality that defines where attach the hook

# FUNCTION TO DECIDE WHERE ATTACH THE HOOK
def attach_hook(counter):
      # TRY MULTIPLE CONFIGURATIONS
      CONV_LAYERS = 15
      FIRST_LAYER = 0
      LAST_LAYER = CONV_LAYERS
      MIDDLE_LAYER = int(CONV_LAYERS/2)
      
      if MODE == 'counter_step':
          return counter % STEP == 0
      elif MODE == 'first':
          return counter == FIRST_LAYER
      elif MODE == 'middle':
          return counter == MIDDLE_LAYER
      elif MODE == 'last':
          return counter == LAST_LAYER
      elif MODE == 'first_middle':
          return counter == FIRST_LAYER or counter == MIDDLE_LAYER
      elif MODE == 'middle_last':
          return counter == MIDDLE_LAYER or counter == LAST_LAYER
      elif MODE == 'first_last':
          return counter == FIRST_LAYER or counter == LAST_LAYER
      else:
          return False

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
        counter = 0
        for name,layer in self.resnet.named_modules():
            # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha 
            if isinstance(layer, nn.Conv2d) and not ('downsample' in name) and not (name == 'conv1'):
                #print(f"name: {name}, layer: {layer}")
                if attach_hook(counter):
                #if name == 'layer4.1.conv2':
                    self.hooks.append(layer.register_forward_hook(self.get_activation_maps_hook))
                    counter+=1
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def attach_apply_activation_maps_hooks(self):
        counter = 0
        for name,layer in self.resnet.named_modules():
            # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha 
            if isinstance(layer, nn.Conv2d) and not ('downsample' in name) and not (name == 'conv1'):
                #print(f"name: {name}, layer: {layer}")
                if attach_hook(counter):
                #if name == 'layer4.1.conv2':
                    self.hooks.append(layer.register_forward_hook(self.apply_activation_maps_hook))
                    counter+=1

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
        counter = 0
        for name,layer in self.resnet.named_modules():
            # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha 
            if isinstance(layer, nn.Conv2d) and not ('downsample' in name) and not (name == 'conv1'):
                #print(f"name: {name}, layer: {layer}")
                if attach_hook(counter):
                #if name == 'layer4.1.conv2':
                    self.hooks.append(layer.register_forward_hook(self.random_activation_maps_hook))
                    counter+=1
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def random_activation_maps_hook(self, module, input, output):
        # Apply the activation shaping function to the source data
        output_A = torch.where(output <= 0, 0.0, 1.0)
        # Specify the desired ratio of 1s (e.g., 0.3 for 30% of 1s)
        # YOU SHOULD TRY DIFFERENT RATIOS ==> CHANGE desired_ratio_of_ones FOR EXPERIMENTING
        desired_ratio_of_ones = RATIO  # STARTING FROM M MADE OF ONLY ONES
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
        counter = 0
        for name,layer in self.resnet.named_modules():
            # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha 
            if isinstance(layer, nn.Conv2d) and not ('downsample' in name) and not (name == 'conv1'):
                #print(f"name: {name}, layer: {layer}")
                if attach_hook(counter):
                #if name == 'layer4.1.conv2':
                    self.hooks.append(layer.register_forward_hook(self.get_activation_maps_hook))
                    counter+=1
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def attach_apply_activation_maps_hooks(self):
        counter = 0
        for name,layer in self.resnet.named_modules():
            # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha 
            if isinstance(layer, nn.Conv2d) and not ('downsample' in name) and not (name == 'conv1'):
                #print(f"name: {name}, layer: {layer}")
                if attach_hook(counter):
                #if name == 'layer4.1.conv2':
                    self.hooks.append(layer.register_forward_hook(self.apply_activation_maps_hook))
                    counter+=1

    def get_activation_maps_hook(self, module, input, output):
        self.activations.append(output.clone().detach())

    def apply_activation_maps_hook(self, module, input, output):
        # Apply the activation shaping function to the source data
        output_A = output
        M = self.activations.pop(0)
        # BINARIZE AND TOP K ONLY ON VARIANT 2
        if self.variation==2:
            M = torch.where(M <= 0, 0.0, 1.0)
            k = K
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
        counter = 0
        for name,layer in self.resnet.named_modules():
            # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha 
            if isinstance(layer, nn.Conv2d) and not ('downsample' in name) and not (name == 'conv1'):
                #print(f"name: {name}, layer: {layer}")
                if attach_hook(counter):
                #if name == 'layer4.1.conv2':
                    self.hooks.append(layer.register_forward_hook(self.random_activation_maps_hook))
                    counter+=1
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def random_activation_maps_hook(self, module, input, output):
        # Apply the activation shaping function to the source data
        output_A = output
        # Specify the desired ratio of 1s (e.g., 0.3 for 30% of 1s)
        # YOU SHOULD TRY DIFFERENT RATIOS ==> CHANGE desired_ratio_of_ones FOR EXPERIMENTING
        desired_ratio_of_ones = RATIO  # STARTING FROM M MADE OF ONLY ONES
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
            k = K
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