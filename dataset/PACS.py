import torch
import os
import torchvision.transforms as T
from dataset.utils import BaseDataset , DomainAdaptationDataset , DomainGeneralizationDataset
from dataset.utils import SeededDataLoader
import random

from globals import CONFIG

def get_transform(size, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(T.Resize(256))
        transform.append(T.RandomResizedCrop(size=size, scale=(0.7, 1.0)))
        transform.append(T.RandomHorizontalFlip())
    else:
        transform.append(T.Resize(size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean, std))
    return T.Compose(transform)

def domain_generalization_setup():
    # final list of (domain_1,domain_2,domain_3,label) pair to be returned
    domains_pairs = []
    # load the three domains files, except for the target one
    domains = []
    for domain in ['photo','sketch','cartoon','art_painting']:
        # load path/label pair only if different than the target domain
        if domain != CONFIG.dataset_args['target_domain']:
            domain_data = []
            with open(os.path.join(CONFIG.dataset_args['root'], f"{domain}.txt"), 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                path, label = line[0].split('/')[1:], int(line[1])
                domain_data.append((os.path.join(CONFIG.dataset_args['root'], *path), label))
            domains.append(domain_data)
    # given a fixed class, get path/label for each domain and create (domain_1,domain_2,domain_3,label) pair
    for Class in range(0,CONFIG.num_classes):
        # filter the pair by the current class
        domain_1 = [pair for pair in domains[0] if pair[1] == Class]
        domain_2 = [pair for pair in domains[1] if pair[1] == Class]
        domain_3 = [pair for pair in domains[2] if pair[1] == Class]
        # get maximum_length among all domains
        domain_1_length = len(domain_1)
        domain_2_length = len(domain_2)
        domain_3_length = len(domain_3)
        maximum_length = max(domain_1_length, domain_2_length, domain_3_length)
        # get a value for each domain, by taking into account his length
        for i in range(0,maximum_length):
            # create (domain_1,domain_2,domain_3,label) pair
            pair = []
            # if ecceds the length take a random element from the domain to fill
            if i < domain_1_length:
                pair.append(domain_1[i][0])
            else:
                pair.append(random.choice(domain_1)[0])
            # repeat for other domains
            if i < domain_2_length:
                pair.append(domain_2[i][0])
            else:
                pair.append(random.choice(domain_2)[0])

            if i < domain_3_length:
                pair.append(domain_3[i][0])
            else:
                pair.append(random.choice(domain_3)[0])
            # append the class label
            pair.append(Class)
            # convert to tuple and add to the final list
            domains_pairs.append(tuple(pair))
    return domains_pairs

def load_data():
    CONFIG.num_classes = 7
    CONFIG.data_input_size = (3, 224, 224)

    # Create transforms
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet Pretrain statistics
    train_transform = get_transform(size=224, mean=mean, std=std, preprocess=True)
    test_transform = get_transform(size=224, mean=mean, std=std, preprocess=False)

    # Load examples & create Dataset
    if CONFIG.experiment in ['baseline']:
        source_examples, target_examples = [], []

        # Load source
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        train_dataset = BaseDataset(source_examples, transform=train_transform, is_test=False)
        test_dataset = BaseDataset(target_examples, transform=test_transform, is_test=True)

    ######################################################
    elif CONFIG.experiment in ['activation_shaping_module','extension_2_activation_shaping_module']:
        source_examples, target_examples = [], []

        # Load source
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        train_dataset = DomainAdaptationDataset(source_examples, target_examples, transform=train_transform)
        test_dataset = BaseDataset(target_examples, transform=test_transform)
    
    ######################################################
    elif CONFIG.experiment in ['random_activation_maps','extension_2_random_activation_maps']:
        source_examples, target_examples = [], []

        # Load source
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        train_dataset = BaseDataset(source_examples, transform=train_transform)
        test_dataset = BaseDataset(target_examples, transform=test_transform)

    elif CONFIG.experiment in ['domain_generalization']:
        source_examples, target_examples = [], []
        # get the domains data and set up the Dataset
        source_examples = domain_generalization_setup()
        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))
        train_dataset = DomainGeneralizationDataset(source_examples, transform=train_transform)
        test_dataset = BaseDataset(target_examples, transform=test_transform,is_test=True)

    ######################################################

    # Dataloaders
    train_loader = SeededDataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = SeededDataLoader(
        test_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return {'train': train_loader, 'test': test_loader}