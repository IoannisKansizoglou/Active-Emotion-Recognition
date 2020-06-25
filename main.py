import torch, sys
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np, json

from extractors import VisualNet, AudioNet, FusionNet
from train import train_model, predict_model
from datasets import get_datasets

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def define_model(modality,speaker):
    if modality=='audio':
        model = AudioNet()
    elif modality=='visual':
        model = VisualNet()
    elif modality=='fusion':
        model = FusionNet(speaker)
    else:
        print("Error[0]: Unknown modality "+str(modality))
    return model


def main(mode, speaker, modality, epochs, batch_size, lr, step_size, gamma):

    dataloaders, dataset_sizes = get_datasets(speaker, modality=modality, batch_size=batch_size)

    model = define_model(modality,speaker)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if mode=='train':
        model = train_model(model, optimizer, exp_lr_scheduler, device, dataloaders, dataset_sizes,
                                    speaker, num_epochs=epochs, modality=modality)
        torch.save(model.state_dict(), 'data/'+speaker+'/'+modality+'/model')
    else:
        print('test time!')
        # features_list, labels_list, preds_list = predict_model(model, device, dataloaders, dataset_sizes)

    pass

# Control runtime
if __name__ == '__main__':
    if sys.argv[1] == 'test':
        mode='test'
    else:
        mode='train'
    with open('params.json') as json_file:
        params = json.load(json_file)
        main(
            mode=mode,
            speaker=params['speaker'], 
            modality=params['modality'], 
            epochs=params['num_epochs'],
            batch_size=params['batch_size'],
            lr = params['learning_rate'],
            step_size=params['decay_step'],
            gamma=params['decay_rate']
        )

    