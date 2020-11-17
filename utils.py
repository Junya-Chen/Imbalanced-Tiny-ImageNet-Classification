import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import os, numpy as np
from TinyNetPretrain import train_model
from torch.utils.data import Dataset, DataLoader



# Split Training dataset into training (size: 450) and validation (size: 50).
# Labels in the original validation set are all zeros.
def split_train_valid(data_dir= 'tiny-imagenet-200', n_sample=50):
    
    data_transforms = transforms.Lambda(lambda image: 
                                        (np.array(image).astype('uint8')).transpose(2,0,1))

    image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                              data_transforms)

    train_loader = DataLoader(image_dataset_train, 
                                                 batch_size=256, num_workers=10,shuffle=False, drop_last=False)

    label_ = []
    x_ = []
    for x, label in train_loader:
        label_.append(label)
        x_.append(x)

    label_ = torch.cat(label_)
    x_ = torch.cat(x_)
    label_list = torch.unique(label_).tolist()


    torch.manual_seed(123)
    valid_idx = []

    for label in label_list:
        label_idx = np.where(label_==label)[0]
        sub_label_idx = np.random.choice(label_idx, n_sample)
        valid_idx.extend(list(sub_label_idx))
        
    valid_x = x_[valid_idx]
    valid_label = label_[valid_idx]

    torch.save(valid_x,data_dir+'/valid_x.pt')
    torch.save(valid_label,data_dir+'/valid_label.pt')

    train_idx = list(set(range(len(label_))) - set(valid_idx))
    train_x = x_[train_idx]
    train_label = label_[train_idx]

    torch.save(train_x, data_dir+'/train_x.pt')
    torch.save(train_label, data_dir+'/train_label.pt')
    
    
# Pretrain
def pretrain(dataloaders, dataset_sizes, criterion, model_path, num_cls=200, num_epochs=500):
    model_ft = models.resnet18(pretrained=True)
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    model_ft.fc.out_features = num_cls
    device = 'cuda'
    model_ft = model_ft.to(device)
    #Multi GPU
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) 
    model_ft = nn.DataParallel(model_ft).cuda()
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs, model_path)

def finetune(dataloaders, dataset_sizes, criterion, model_path, ft_model_path, num_epochs=200, num_cls=200):
    # Finetune
    # Load Resnet18
    device = 'cuda'
    model_ft = models.resnet18()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) 
    #Finetune Final few layers to adjust for tiny imagenet input
    model_ft.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model_ft.maxpool = nn.Sequential()
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    model_ft.fc.out_features = num_cls
    model_ft = model_ft.to(device)
    model_ft = nn.DataParallel(model_ft).cuda()
    pretrained_dict = torch.load(model_path)
    model_ft_dict = model_ft.state_dict()
    first_layer_weight = model_ft_dict['module.conv1.weight']
    pretrained_dict = {b[0]:b[1] for a,b in zip(model_ft_dict.items(), pretrained_dict.items()) if a[1].size() == b[1].size()}
    model_ft_dict.update(pretrained_dict) 
    model_ft.load_state_dict(model_ft_dict)
    #Train
    train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs, ft_model_path)
    
def check_ft_model(valid_loader, model_ft, device='cuda'):
    check_precision = 0
    for i,(x,labels) in enumerate(valid_loader):
        model_ft.eval()
        x = x.to(device)
        labels = labels.to(device)
        outputs = model_ft(x)
        _, preds = torch.max(outputs, 1)
        check_precision += torch.sum(preds == labels.data).item()
    valid_acc = check_precision/len(valid_loader.dataset.targets)
    print('Validation Acc of Final Model: {:.4f}%'.format(valid_acc*100))

    
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
# Extract features from final model (if needed)
# Pop the last layer (fc = Identity())   
def save_batched_features(imb_type, imb_factor, train_loader, valid_loader, model_ft):
    model_ft.module.fc = Identity()
    model_ft.eval()
    train_save = 'train_'+imb_type+"_("+str(imb_factor)+")"+'_batch/'
    valid_save = 'valid_'+imb_type+"_("+str(imb_factor)+")"+'_batch/'

    for batch_idx, samples in enumerate(train_loader):   
        torch.save(model_ft(samples[0].to(device)), train_save+'batch'+'_'+str(batch_idx)+'_'+'x.pt')
        torch.save(samples[1], train_save+'batch'+'_'+str(batch_idx)+'_'+'label.pt')    

    for batch_idx, samples in enumerate(valid_loader):   
        torch.save(model_ft(samples[0].to(device)), valid_save+'batch'+'_'+str(batch_idx)+'_'+'x.pt')
        torch.save(samples[1], valid_save+'batch'+'_'+str(batch_idx)+'_'+'label.pt')