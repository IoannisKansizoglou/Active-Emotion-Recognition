import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from vggish import VGGish

import numpy as np

class VisualNet(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(VisualNet, self).__init__()
        self.num_classes = num_classes

        self.extractor = models.mobilenet_v2(pretrained=pretrained)
        self.num_ftrs = self.extractor.classifier[1].in_features
        self.extractor.classifier = self.extractor.classifier[0]
        self.linear = nn.Linear(self.num_ftrs,num_classes)
        
    def forward(self, x, labels):
        embds = self.extractor(x)
        out = self.linear(embds)
        return out, embds

class AudioNet(nn.Module):
    def __init__(self, num_classes=6):
        super(AudioNet, self).__init__()
        self.num_classes = num_classes
        
        self.extractor = VGGish()
        self.num_ftrs = self.extractor.fc[4].out_features
        self.extractor.load_state_dict(torch.load('data/weights/pytorch_vggish.pth'))
        self.linear = nn.Linear(self.num_ftrs,num_classes)

    def forward(self, x, labels):
        embds = self.extractor(x)
        out = self.linear(embds)
        return out,embds

class FusionNet(nn.Module):
    def __init__(self, speaker, num_classes=6, fine_tune=False):
        super(FusionNet, self).__init__()
        self.num_classes = num_classes

        self.visual_extractor = VisualNet(pretrained=False)
        self.audio_extractor = AudioNet()
        self.load_weights(speaker)
        if not fine_tune:
            for param in self.visual_extractor.parameters():
                param.requires_grad = False
            for param in self.audio_extractor.parameters():
                param.requires_grad = False
        self.bn1 = nn.BatchNorm1d(1280+128)
        self.linear1 = nn.Linear(1280+128,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512,512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512,num_classes)

    def load_weights(self, speaker):
        self.visual_extractor.load_state_dict(torch.load('data/'+speaker+'/visual/model'))
        self.audio_extractor.load_state_dict(torch.load('data/'+speaker+'/audio/model'))
        
        pass

    def forward(self, image, spec, labels):
        _,visual_features = self.visual_extractor(image,labels)
        _,audio_features = self.audio_extractor(spec,labels)
        out = self.bn1(torch.cat((visual_features,audio_features),-1))
        out = F.relu(self.bn2(self.linear1(out)))
        out = F.relu(self.bn3(self.linear2(out)))
        out = self.linear3(out)
        
        return out

