from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torch.optim.lr_scheduler import StepLR

from wide_resnet import Wide_ResNet
import plot

torch.manual_seed(0)

batch_size = 256
use_cuda = False
device = 'cpu'
"""kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        layers = OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 3, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('pool1', nn.MaxPool2d(2)),
            ('dropout1', nn.Dropout2d(0.25)),
            ('flatten', nn.Flatten()),
            ('dense1', nn.Linear(9216, 128)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout2d(0.5)),
            ('dense2', nn.Linear(128, 10)),
            ('output', nn.LogSoftmax(dim=1))
        ])
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        output = self.layers(x)
        return output

def gradient_quotient(loss, params, eps=1e-5):
    # Calculate g(theta) = gradient of loss
    grad = torch.autograd.grad(
            loss,
            params,
            retain_graph=True, create_graph=True) # <= required for higher-order gradients

    # Calculate H(theta) * g(theta)
    prod = torch.autograd.grad(
            sum([(g**2).sum() / 2 for g in grad]),
            params,
            retain_graph=True, create_graph=True)

    # Form the gradient quotient (GQ) loss as in eq (1)
    out = sum([((g - p) / (g + eps * (2*(g >= 0).float() - 1).detach())- 1).abs().sum() \
                for g, p in zip(grad, prod)])

    return out / sum([p.nelement() for p in params])

def metainit(model, criterion, x_size, y_size, lr=0.01, momentum=0.5, steps=100, eps=1e-5):
    model.eval()

    # Only perform gradient on matrix and tensor parameters, not vector biases
    params = [p for p in model.parameters() if p.requires_grad and len(p.size()) >= 2]
    
    # Exponential moving average of first moment to implement momentum
    memory = [0] * len(params)

    for i in range(steps):
        # Draw input i.i.d. from N[0,1]
        input = torch.Tensor(*x_size).normal_(0, 1).cuda()

        # Draw output i.i.d. from uniform over integers
        target = torch.randint(0, y_size, (x_size[0],)).cuda()

        # Calculate learning and meta-learning losses
        y = model(input)
        loss = criterion(y, target)
        gq = gradient_quotient(loss, list(model.parameters()), eps)
        
        # For each parameter matrix/tensor, perform step of signSGD
        grad = torch.autograd.grad(gq, params)
        for j, (p, g_all) in enumerate(zip(params, grad)):
            # Calculate L_2/Frobenius norm of parameter
            norm = p.norm().item()

            # d(GQ)/d(norm of parameter)
            g = torch.sign((p * g_all).sum() / norm)

            # signSGD update rule
            memory[j] = momentum * memory[j] - lr * g.item()
            new_norm = norm + memory[j]
            
            # Update norm of parameter
            #print(new_norm / norm)
            p.detach().mul_(new_norm / norm)
        
        print("%d/GQ = %.2f" % (i, gq.item()))

# Basical model
device = 'cuda'
#model = Net().to(device)

model = Wide_ResNet(28, 10, 0.3, 10).cuda()
#y = model(torch.randn(1,3,32,32))

#for idx, (n, p) in enumerate(model.named_parameters()):
#    print(idx, n)

"""for batch_idx, (data, target) in enumerate(train_loader):
    print(data.size(), target.size())
    break"""

#def weight_norms(model):
#    norms = [(n,p.norm().item()) for n,p in model.named_parameters() if p.requires_grad and len(p.size()) >= 2]
#    return norms

#print(weight_norms(model))

# MetaInit for basic network on MNIST sized image
#metainit(model, F.nll_loss, (256, 1, 28, 28), 10)

def weight_norms(model):
    norms = [(n,p.norm().item()) for n,p in model.named_parameters() if p.requires_grad and len(p.size()) >= 2]
    norms = [np.array(n) for _,n in norms]
    return norms

def scale_weights(model, scale=1/4):
    for p in model.parameters():
        if p.requires_grad and len(p.size()) >= 2:
            p.detach().mul_(scale)

wns_xavier = weight_norms(model)
scale_weights(model)
wns_before = weight_norms(model)

#print(len(wns))

# MetaInit on basic CNN for Cifar-10
metainit(model, F.nll_loss, (8, 3, 32, 32), 10)
wns_after = weight_norms(model)

#wns = [n for _,n in wns]
plot.plot_weights([('xavier', wns_xavier), ('before', wns_before), ('after', wns_after)])