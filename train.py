import torch, time, copy
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def train_model(model, optimizer, scheduler, device, dataloaders, dataset_sizes, speaker,
                target_path='data/', num_epochs=25, modality='rnd'):
    
    writer = SummaryWriter(target_path+speaker+'/'+modality+'/')
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{} with lr = {}'.format(epoch, num_epochs - 1,optimizer.param_groups[0]['lr']))
        print('-' * 25)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for sample in dataloaders[phase]:                
                inputs = sample['image']
                labels = sample['label']
                inputs = inputs.to(device, torch.float)
                labels = labels.to(device)
                if modality=='fusion':
                    specs = sample['spec']
                    specs = specs.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if modality=='fusion':
                        outputs = model(inputs,specs,labels)
                        loss = nn.CrossEntropyLoss()(outputs,labels)                      
                    else:
                        outputs,_ = model(inputs,labels)
                        loss = nn.CrossEntropyLoss()(outputs,labels)
                        
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))            
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                writer.add_scalar('Loss/test', epoch_loss, epoch)
                writer.add_scalar('Accuracy/test', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    return model

def predict_model(model, device, dataloaders, dataset_sizes, phase='val'):
    features_list, labels_list, preds_list = list(), list(), list()
    for sample in dataloaders[phase]:
        with torch.no_grad():
            inputs = sample['image']
            labels = sample['label']
            inputs = inputs.to(device, torch.float)
            labels = labels.to(device)
            outputs,features = model(inputs,labels)
            
            _, preds = torch.max(outputs, 1)
            features_list.extend(features.to('cpu'))
            labels_list.extend(labels.to('cpu'))
            preds_list.extend(preds.to('cpu', np.int))
    features_list = np.array([np.array(features) for features in features_list])
    labels_list = np.array(labels_list)
    preds_list = np.array(preds_list)

    return features_list, labels_list, preds_list