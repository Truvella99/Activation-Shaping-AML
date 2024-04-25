import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm

import os
import logging
import warnings
import random
import numpy as np
from parse_args import parse_arguments

from dataset import PACS
from models.resnet import BaseResNet18,ASHResNet18,RAMResNet18,EXTASHResNet18,EXTRAMResNet18,DOMGENResNet18,LAYERS

from globals import CONFIG

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    
    acc_meter = Accuracy(task='multiclass', num_classes=CONFIG.num_classes)
    acc_meter = acc_meter.to(CONFIG.device)

    loss = [0.0, 0]
    for x, y,*_ in tqdm(data):
        with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
            x, y = x.to(CONFIG.device), y.to(CONFIG.device)
            logits = model(x)
            acc_meter.update(logits, y)
            loss[0] += F.cross_entropy(logits, y).item()
            loss[1] += x.size(0)
    
    accuracy = acc_meter.compute()
    loss = loss[0] / loss[1]
    logging.info(f'Accuracy: {100 * accuracy:.2f} - Loss: {loss}')


def train(model, data):

    # Create optimizers & schedulers
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0005, momentum=0.9, nesterov=True, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG.epochs * 0.8), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Load checkpoint (if it exists)
    cur_epoch = 0
    if os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
        checkpoint = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
    
    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(data['train'])):
            
            # Call separately the model to firstly record the activation maps
            # This only needed for activation_shaping_module and extension_2_activation_shaping_module
            if CONFIG.experiment in ['activation_shaping_module','extension_2_activation_shaping_module']:
                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    # set model to eval() to record the activations without influencing the following training
                    model.eval()
                    src_x, src_y, target_x = batch
                    src_x, src_y, target_x = src_x.to(CONFIG.device), src_y.to(CONFIG.device), target_x.to(CONFIG.device)
                    # attach the hook to record the activation maps
                    model.attach_get_activation_maps_hooks()
                    # perform the forward pass without keeping track of the gradient
                    with torch.no_grad():
                        model(target_x)
                    # remove the previously attached hooks and set the model back to train()
                    model.remove_hooks()
                    model.train()
            elif CONFIG.experiment in ['domain_generalization']:
                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    # set model to eval() to record the activations without influencing the following training
                    model.eval()
                    src_x1, src_x2, src_x3, src_y = batch
                    src_x1, src_x2, src_x3, src_y = src_x1.to(CONFIG.device), src_x2.to(CONFIG.device), src_x3.to(CONFIG.device), src_y.to(CONFIG.device)
                    # attach the hook to record the activation maps of the three domains
                    model.attach_get_activation_maps_hooks()
                    # perform the forward pass for the three domains without keeping track of the gradient
                    with torch.no_grad():
                        model(src_x1)
                        model(src_x2)
                        model(src_x3)
                    # remove the previously attached hooks and set the model back to train()
                    model.remove_hooks()
                    model.train()

            # Compute loss
            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):

                if CONFIG.experiment in ['baseline']:
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)
                elif CONFIG.experiment in ['activation_shaping_module','extension_2_activation_shaping_module']:
                    src_x, src_y, target_x = batch
                    src_x, src_y, target_x = src_x.to(CONFIG.device), src_y.to(CONFIG.device), target_x.to(CONFIG.device)
                    # attach the activation shaping hook
                    model.attach_apply_activation_maps_hooks()
                    # perform the forward pass/compute the cross-entropy loss
                    loss = F.cross_entropy(model(src_x), src_y)
                    # remove the previously attached hooks
                    model.remove_hooks()
                elif CONFIG.experiment in ['random_activation_maps','extension_2_random_activation_maps']:
                    src_x, src_y = batch
                    src_x, src_y = src_x.to(CONFIG.device), src_y.to(CONFIG.device)
                    # attach the random activation maps hook
                    model.attach_random_activation_maps_hooks()
                    # perform the forward pass/compute the cross-entropy loss
                    loss = F.cross_entropy(model(src_x), src_y)
                    # remove the previously attached hooks
                    model.remove_hooks()
                elif CONFIG.experiment in ['domain_generalization']:
                    src_x1, src_x2, src_x3, src_y = batch
                    src_x1, src_x2, src_x3, src_y = src_x1.to(CONFIG.device), src_x2.to(CONFIG.device), src_x3.to(CONFIG.device), src_y.to(CONFIG.device)
                    # attach the activation shaping hook
                    model.attach_apply_activation_maps_hooks()
                    # perform the forward pass/compute the cross-entropy loss
                    # by concatenating in a single minibatch both src_x and src_y (3 times src_y to match the dimension)
                    loss = F.cross_entropy(model(torch.cat((src_x1, src_x2, src_x3), dim=0)), torch.cat((src_y,src_y,src_y), dim=0))
                    # remove the previously attached hooks
                    model.remove_hooks()

            # Optimization step
            scaler.scale(loss / CONFIG.grad_accum_steps).backward()

            if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                scaler.step(optimizer)
                # optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

        scheduler.step()
        
        # Test current epoch
        logging.info(f'[TEST @ Epoch={epoch}]')
        evaluate(model, data['test'])

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join('record', CONFIG.experiment_name, 'last.pth'))


def main():
    
    # Load dataset
    data = PACS.load_data()

    # Load model
    if CONFIG.experiment in ['baseline']:
        model = BaseResNet18()
    elif CONFIG.experiment in ['activation_shaping_module']:
        model = ASHResNet18()
    elif CONFIG.experiment in ['random_activation_maps']:
        model = RAMResNet18()
    elif CONFIG.experiment in ['domain_generalization']:
        model = DOMGENResNet18()
    elif CONFIG.experiment in ['extension_2_activation_shaping_module']:
        model = EXTASHResNet18(variation=1)
    elif CONFIG.experiment in ['extension_2_random_activation_maps']:
        model = EXTRAMResNet18(variation=1)
    ######################################################
    #elif... TODO: Add here model loading for the other experiments (eg. DA and optionally DG)

    ######################################################
    
    model.to(CONFIG.device)

    if not CONFIG.test_only:
        train(model, data)
    else:
        evaluate(model, data['test'])
    

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)

    # Parse arguments
    args = parse_arguments()
    CONFIG.update(vars(args))

    # Setup output directory
    CONFIG.save_dir = os.path.join('record', CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)

    # Setup logging file name based on the layers used
    log_name = 'log_layers__' + '_'.join([key.replace('layer', '') for key, value in LAYERS.items() if value]) + '.txt'

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, log_name), 
        format='%(message)s', 
        level=logging.INFO, 
        filemode='a'
    )

    # Delete last.pth file if it exists
    if os.path.exists(os.path.join(CONFIG.save_dir, 'last.pth')):
        # Delete the file
        os.remove(os.path.join(CONFIG.save_dir, 'last.pth'))
    
    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')

    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    main()
