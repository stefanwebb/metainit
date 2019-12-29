import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import plot

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.Identity() # nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Identity() #nn.Dropout(p=dropout_rate)
        self.bn2 = nn.Identity() # nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        #self.shortcut = nn.Sequential()
        #if stride != 1 or in_planes != planes:
        #    self.shortcut = nn.Sequential(
        #        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
        #    )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        #out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'

        # Number of layers per section
        n = (depth-4)//6

        # TODO: Intuitive description
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        #self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.bn1 = nn.Identity()
        self.linear = nn.Linear(nStages[3], num_classes)
        #self.output_fn = nn.LogSoftmax(dim=1)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # (1, 3, 32, 32)
        out = self.conv1(x)

        # (1, 16, 32, 32)
        out = self.layer1(out)

        # (1, 160, 32, 32)
        # NOTE: Goes from 16 to 160 feature maps because widen factor is 10
        out = self.layer2(out)

        # (1, 320, 16, 16)
        out = self.layer3(out)

        # (1, 640, 8, 8)
        out = F.relu(self.bn1(out)) 

        # (1, 640, 8, 8)
        out = F.avg_pool2d(out, 8)

        # (1, 640, 1, 1)
        out = out.view(out.size(0), -1)

        # (1, 640)
        out = self.linear(out) 

        # (1, 10)
        return F.log_softmax(out, dim=1)

if __name__ == '__main__':
    model = Wide_ResNet(28, 10, 0.3, 10)
    y = model(torch.randn(1,3,32,32))

    #for n, p in model.named_parameters():
    #    print(n)

    def weight_norms(model):
        norms = [(n,p.norm().item()) for n,p in model.named_parameters() if p.requires_grad and len(p.size()) >= 2]
        return norms

    wns = weight_norms(model)
    print(len(wns))

    wns = [n for _,n in wns]
    plot.plot_weights(np.array(wns))

    #print(y.size())