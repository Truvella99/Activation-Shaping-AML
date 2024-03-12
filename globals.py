import torch
import torch.backends.mps

###############################################
#                  FLAGS
###############################################
# Flag to decide if apply data augmentation, set to True for apply it False otherwise
DATA_AUGMENTATION = False
# Factor that tell us how much we are data augmentating the data by generating new ones, only if DATA_AUGMENTATION set to True
DATA_AUGMENTATION_FACTOR = 2.0
# Flag to decide if run in local, set to True for run locally False otherwise
RUN_LOCALLY = False

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

CONFIG = dotdict({})

if torch.cuda.is_available():
    CONFIG.device = 'cuda'
elif torch.backends.mps.is_available() and \
    torch.backends.mps.is_built():
    CONFIG.device = 'mps'
else:
    CONFIG.device = 'cpu'

CONFIG.dtype = torch.float32