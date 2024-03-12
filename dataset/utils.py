import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

import numpy as np
import random
from PIL import Image

from globals import CONFIG,RUN_LOCALLY,DATA_AUGMENTATION,DATA_AUGMENTATION_FACTOR

def get_data_augmentation_transformations():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet Pretrain statistics
    data_augmentation_techniques = []
    n_of_techniques = 5
    # Choose n_of_techniques random data_augmentation_techniques from the list (10 techniques)
    # Actually they will be n_of_techniques + 1 (final random resize at the end to ensure same size)
    random_techniques = random.sample([i for i in range(10)],n_of_techniques)
    # append them to the list
    for random_technique in random_techniques:
        match random_technique:
            case 0:
                data_augmentation_techniques.append(T.RandomHorizontalFlip(p=0.5))
            case 1:
                data_augmentation_techniques.append(T.RandomVerticalFlip(p=0.5))
            case 2:
                data_augmentation_techniques.append(T.RandomRotation(degrees=(-45, 45)))
            case 3:
                data_augmentation_techniques.append(T.RandomRotation(degrees=(-45, 45)))
            case 4:
                data_augmentation_techniques.append(T.ColorJitter(brightness=0.2))
            case 5:
                data_augmentation_techniques.append(T.ColorJitter(contrast=0.2))
            case 6:
                data_augmentation_techniques.append(T.ColorJitter(saturation=0.2))
            case 7:
                data_augmentation_techniques.append(T.ColorJitter(hue=0.1))
            case 8:
                data_augmentation_techniques.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
            case 9:
                data_augmentation_techniques.append(T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10))
    # resize to have same size of test samples (+1), convert to tensor and normalize
    data_augmentation_techniques.append(T.RandomResizedCrop(size=(224, 224), scale=(0.5, 2.0)))
    data_augmentation_techniques.append(T.ToTensor())
    data_augmentation_techniques.append(T.Normalize(mean, std))
    # return them
    return T.Compose(data_augmentation_techniques)

class BaseDataset(Dataset):
    def __init__(self, examples, transform, is_test):
        self.examples = examples
        self.len_examples = len(examples)
        self.T = transform
        # to understand if is test dataset, in that case not apply data augmentation
        # needed only for this dataset class, since only used also for test samples
        self.is_test = is_test
    
    def __len__(self):
        if self.is_test or (not DATA_AUGMENTATION):
            return self.len_examples
        else:
            # dataset items + augmented items to generate (how many depends on the DATA_AUGMENTATION_FACTOR)
            return int(self.len_examples * DATA_AUGMENTATION_FACTOR)
    
    def __getitem__(self, index):
        if self.is_test or (not DATA_AUGMENTATION) or (DATA_AUGMENTATION and index < self.len_examples):
            x, y = self.examples[index]
            x = Image.open(x).convert('RGB')
            x = self.T(x).to(CONFIG.dtype)
            y = torch.tensor(y).long()
            return x, y
        else:
            # apply data augmentation and finished original dataset items case, so generate new data
            x, y = random.choice(self.examples)
            x = Image.open(x).convert('RGB')
            x = get_data_augmentation_transformations()(x).to(CONFIG.dtype)
            y = torch.tensor(y).long()
            return x, y

class DomainAdaptationDataset(Dataset):
    def __init__(self, source_examples, target_examples, transform):
        self.source_examples = source_examples
        self.len_source_examples = len(source_examples)
        self.target_examples = target_examples
        self.T = transform
    
    def __len__(self):
        if not DATA_AUGMENTATION:
            return self.len_source_examples
        else:
            # dataset items + augmented items to generate (how many depends on the DATA_AUGMENTATION_FACTOR)
            return int(self.len_source_examples * DATA_AUGMENTATION_FACTOR)
    
    def __getitem__(self, index):
        if (not DATA_AUGMENTATION) or (DATA_AUGMENTATION and index < self.len_source_examples):
            src_x, src_y = self.source_examples[index][0] , self.source_examples[index][1] 
            # Randomly sample a target example
            targ_x,_  = random.choice(self.target_examples)
            src_x = Image.open(src_x).convert('RGB')
            src_x = self.T(src_x).to(CONFIG.dtype)
            src_y = torch.tensor(src_y).long()
            
            targ_x = Image.open(targ_x).convert('RGB')
            targ_x = self.T(targ_x).to(CONFIG.dtype)

            return src_x, src_y, targ_x
        else:
            # apply data augmentation and finished original dataset items case, so generate new data
            src_x, src_y = random.choice(self.source_examples) 
            # Randomly sample a target example
            targ_x,_  = random.choice(self.target_examples)
            src_x = Image.open(src_x).convert('RGB')
            src_x = get_data_augmentation_transformations()(src_x).to(CONFIG.dtype)
            src_y = torch.tensor(src_y).long()
            
            targ_x = Image.open(targ_x).convert('RGB')
            targ_x = get_data_augmentation_transformations()(targ_x).to(CONFIG.dtype)

            return src_x, src_y, targ_x

class DomainGeneralizationDataset(Dataset):
   def __init__(self, examples, transform):
        self.examples = examples
        self.len_examples = len(examples)
        self.T = transform
   
   def __len__(self):
        if not DATA_AUGMENTATION:
            return self.len_examples
        else:
            # dataset items + augmented items to generate (how many depends on the DATA_AUGMENTATION_FACTOR)
            return int(self.len_examples * DATA_AUGMENTATION_FACTOR)
   
   def __getitem__(self, index):
        if (not DATA_AUGMENTATION) or (DATA_AUGMENTATION and index < self.len_examples):
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
        else:
            # apply data augmentation and finished original dataset items case, so generate new data
            x1, x2, x3, y = random.choice(self.examples)
            # convert samples to image and label to tensor
            x1 = Image.open(x1).convert('RGB')
            x1 = get_data_augmentation_transformations()(x1).to(CONFIG.dtype)
            x2 = Image.open(x2).convert('RGB')
            x2 = get_data_augmentation_transformations()(x2).to(CONFIG.dtype)
            x3 = Image.open(x3).convert('RGB')
            x3 = get_data_augmentation_transformations()(x3).to(CONFIG.dtype)
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

