import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

import numpy as np
import random
from PIL import Image

from globals import CONFIG
from parse_args import RUN_LOCALLY

class BaseDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.T = transform
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        x, y = self.examples[index]
        x = Image.open(x).convert('RGB')
        x = self.T(x).to(CONFIG.dtype)
        y = torch.tensor(y).long()
        return x, y

class DomainAdaptationDataset(Dataset):
    def __init__(self, source_examples, target_examples, transform):
        self.source_examples = source_examples
        self.target_examples = target_examples
        self.T = transform
    
    def __len__(self):
        return len(self.source_examples)
    
    def __getitem__(self, index):
        # TO HANDLE ALSO THE CASE IN WHICH THE TARGET IS BIGGER THAN THE SOURCE
        src_x, src_y = self.source_examples[index][0] , self.source_examples[index][1] 
        # Randomly sample a target example
        targ_x,_  = random.choice(self.target_examples)
        src_x = Image.open(src_x).convert('RGB')
        src_x = self.T(src_x).to(CONFIG.dtype)
        src_y = torch.tensor(src_y).long()
        
        targ_x = Image.open(targ_x).convert('RGB')
        targ_x = self.T(targ_x).to(CONFIG.dtype)

        return src_x, src_y, targ_x
      

class DomainGeneralizationDataset(Dataset):
   def __init__(self, examples, transform):
       self.examples = examples
       self.T = transform
   
   def __len__(self):
       return len(self.examples)
   
   def __getitem__(self, index):
       # get the data
       x1, x2, x3, y = self.examples[index]
       # convert samples to image and label to tensor
       x1 = Image.open(x1).convert('RGB')
       x1 = self.T(x1).to(CONFIG.dtype)
       x2 = Image.open(x2).convert('RGB')
       x2 = self.T(x2).to(CONFIG.dtype)
       x3 = Image.open(x3).convert('RGB')
       x3 = self.T(x3).to(CONFIG.dtype)
       y = torch.tensor(y).long()
       # return them
       return x1, x2, x3, y

######################################################

class SeededDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=None, 
                 sampler=None, 
                 batch_sampler=None, 
                 num_workers=0, collate_fn=None, 
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None, 
                 generator=None, *, prefetch_factor=None, persistent_workers=False, 
                 pin_memory_device=""):
        
        if not CONFIG.use_nondeterministic:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            generator = torch.Generator()
            generator.manual_seed(CONFIG.seed)

            worker_init_fn = None if RUN_LOCALLY else seed_worker
        
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, 
                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, 
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, 
                         pin_memory_device=pin_memory_device)

