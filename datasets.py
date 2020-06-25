import numpy as np
import pandas as pd
import torch
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
from augmentation_tools import Rescale, ToTensor, Normal
from skimage import io, transform
from speech_augmentation import augment_spectrogram

class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_paths = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_paths.iloc[idx, 1]
        image = io.imread(img_name)
        label = np.array(self.data_paths.iloc[idx, 3]).astype(np.int)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, transform_flag=True):
        self.data_paths = pd.read_csv(csv_file)
        self.transform_flag = transform_flag
        
    def npy_loader(self, path):
        sample = torch.from_numpy(np.load(path))
        return sample

    def transform(self, sample):
        sample['image'] = augment_spectrogram(sample['image'],1).unsqueeze(0)        
        return sample
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spec_name = self.data_paths.iloc[idx, 2]
        spec = self.npy_loader(spec_name)
        label = np.array(self.data_paths.iloc[idx, 3]).astype(np.int)
        sample = {'image': spec, 'label': label}
        if self.transform_flag:
            sample = self.transform(sample)

        return sample

class FusionDataset(FaceLandmarksDataset,SpectrogramDataset):
    def __init__(self, csv_file, transform=None, transform_flag=True):
        self.visDataset = FaceLandmarksDataset(csv_file, transform)
        self.audDataset = SpectrogramDataset(csv_file, transform_flag)
        self.data_paths = pd.read_csv(csv_file)
        
    def __getitem__(self, idx):
        vis_sample = self.visDataset.__getitem__(idx)
        aud_sample = self.audDataset.__getitem__(idx)
        sample = {
            'image': vis_sample['image'],
            'spec': aud_sample['image'],
            'label': vis_sample['label']
        }
        return sample

def get_datasets(speaker, modality='visual', batch_size=32, root_path='data/', shuffle_flag=True):

    if modality=='visual':
        datasets = {
            'train': FaceLandmarksDataset(csv_file=root_path+speaker+'/training_data.csv',
                                           transform=transforms.Compose([Rescale((224,224)), ToTensor(), Normal()])),
            'val': FaceLandmarksDataset(csv_file=root_path+speaker+'/evaluation_data.csv',
                                           transform=transforms.Compose([Rescale((224,224)),ToTensor(),Normal()]))
        }
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x],batch_size=batch_size,shuffle=shuffle_flag,num_workers=2) for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

        return dataloaders, dataset_sizes
    
    elif modality=='audio':
        datasets = {
            'train': SpectrogramDataset(csv_file=root_path+speaker+'/training_data.csv',transform_flag=True),
            'val': SpectrogramDataset(csv_file=root_path+speaker+'/evaluation_data.csv',transform_flag=True)
        }
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=shuffle_flag, num_workers=2) for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

        return dataloaders, dataset_sizes

    elif modality=='fusion':
        datasets = {
            'train': FusionDataset(csv_file=root_path+speaker+'/training_data.csv',
                                           transform=transforms.Compose([Rescale((224,224)),ToTensor(),Normal()]),transform_flag=True),
            'val': FusionDataset(csv_file=root_path+speaker+'/evaluation_data.csv',
                                           transform=transforms.Compose([Rescale((224,224)),ToTensor(),Normal()]),transform_flag=True)
        }
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=shuffle_flag, num_workers=2) for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

        return dataloaders, dataset_sizes

    else:
        print('Error[0]: Unknown modality type '+modality)
        pass


def get_v_datasets(speaker, modality='visual', batch_size=32, data_path='data/', shuffle_flag=True):

    if modality=='visual':
        datasets = {
            'val': FaceLandmarksDataset(csv_file=data_path, transform=transforms.Compose([Rescale((224,224)),ToTensor(),Normal()]))
        }
    elif modality=='audio':
        datasets = {
            'val': SpectrogramDataset(csv_file=data_path, transform_flag=True)
        }
    elif modality=='fusion':
        datasets = {
            'val': FusionDataset(csv_file=data_path, transform=transforms.Compose([Rescale((224,224)),ToTensor(),Normal()]),transform_flag=True)
        }
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=shuffle_flag, num_workers=2) for x in ['val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['val']}

    return dataloaders, dataset_sizes
