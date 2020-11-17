import torch, torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.models as models
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from Imbalanced_dataset import ImbalanceDataset, SimpleDataset

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pathlib import Path

from utils import split_train_valid, pretrain, finetune, check_ft_model, save_batched_features



if not os.path.exists('trained_models'):
    os.makedirs('trained_models')
    
    
data_dir = 'tiny-imagenet-200'

split_train_valid(data_dir= data_dir, n_sample=50)

train_x = torch.load(data_dir+'/train_x.pt')
train_label = torch.load(data_dir+'/train_label.pt')
valid_x = torch.load(data_dir+'/valid_x.pt')
valid_label = torch.load(data_dir+'/valid_label.pt')


data_transforms = {
     'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
}

imb_type ='step'
imb_factor=.01
device='cuda'

train_ds = ImbalanceDataset(train_x, train_label, imb_type=imb_type, imb_factor=imb_factor, \
                         transform=data_transforms['train'] )
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, num_workers=10, shuffle=True, drop_last=False)
valid_ds = SimpleDataset(valid_x, valid_label, transform=data_transforms['val'])
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=256, num_workers=10, shuffle=True, drop_last=False)

dataloaders = {'train':train_loader, 'val': valid_loader}
dataset_sizes = {'train':len(train_ds.data), 'val': len(valid_ds.data)}

criterion = nn.CrossEntropyLoss()
model_path = "trained_models/TinyImageNet_"+imb_type+"_("+str(imb_factor)+")"+"_resnet18.pt"
ft_model_path = "trained_models/TinyImageNet_"+imb_type+"_("+str(imb_factor)+")"+"_resnet18_ft.pt"

pretrain(dataloaders = dataloaders, dataset_sizes=dataset_sizes, criterion=criterion, model_path=model_path)
finetune(dataloaders = dataloaders, dataset_sizes=dataset_sizes, criterion=criterion, model_path=model_path, ft_model_path=ft_model_path)


#Load Resnet18
model_ft = models.resnet18()
#Finetune Final few layers to adjust for tiny imagenet input
model_ft.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model_ft.maxpool = nn.Sequential()
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
model_ft.fc.out_features = 200
model_ft = model_ft.to(device)
model_ft = torch.nn.DataParallel(model_ft).cuda() 
model_ft.load_state_dict(torch.load(ft_model_path), strict=False)


check_ft_model(valid_loader, model_ft)
save_batched_features(imb_type, imb_factor, train_loader, valid_loader, model_ft)
