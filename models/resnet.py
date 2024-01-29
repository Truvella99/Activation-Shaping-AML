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
RATIO = 1.0
K = 5

###############################################
#                  POINT 1/3
###############################################


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
                # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha 
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
                # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha
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


###############################################
#                  POINT 2
###############################################


class RAMResNet18(nn.Module):
    def __init__(self):
        super(RAMResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def random_m_hook(self, module, input, output):
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
    
    def forward(self, x, Train=None):
        print('\nForward...')

        hooks = []
        counter = 0
        step = 1 # 1 = ALL CONV2D LAYERS

        if Train:  # Sono in train
            for layer in self.resnet.modules():
                if isinstance(layer, nn.Conv2d):
                  if self.attach_hook(mode='counter_step',counter=counter,step=step):
                    hooks.append(layer.register_forward_hook(self.random_m_hook))
                  counter+=1

            src_logits = self.resnet(x)

            for h in hooks:
                h.remove()

            return src_logits
        else: # Sono in Test, x che mi viene passato qui è quello del test quindi target domain
            return self.resnet(x) # THEN YOU SHOULD TEST THE MODEL ON THE TARGET DOMAINS    

###############################################
#                  EXTENSION 2
###############################################
        
class EXTASHResNet18(nn.Module):
    def __init__(self,variation = None):
        super(EXTASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.activations = []
        self.variation = variation

    def hook1(self, module, input, output):
        # print('Forward hook running...')
        self.activations.append(output.clone().detach())

    def hook2(self, module, input, output):
        # Apply the activation shaping function to the source data
        # print('Backward hook running...')
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
                # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha 
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
                # Incremento counter dopo secondo if, parto dal primo. Se incremento counter prima secondo if parto dal primo che matcha
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


class EXTRAMResNet18(nn.Module):
    def __init__(self,variation=None):
        super(EXTRAMResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.variation = variation

    def random_m_hook(self, module, input, output):
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
    
    def forward(self, x, Train=None):
        print('\nForward...')

        hooks = []
        counter = 0
        step = 1 # 1 = ALL CONV2D LAYERS

        if Train:  # Sono in train
            for layer in self.resnet.modules():
                if isinstance(layer, nn.Conv2d):
                  if self.attach_hook(mode='counter_step',counter=counter,step=step):
                    hooks.append(layer.register_forward_hook(self.random_m_hook))
                  counter+=1

            src_logits = self.resnet(x)

            for h in hooks:
                h.remove()

            return src_logits
        else: # Sono in Test, x che mi viene passato qui è quello del test quindi target domain
            return self.resnet(x) # THEN YOU SHOULD TEST THE MODEL ON THE TARGET DOMAINS    