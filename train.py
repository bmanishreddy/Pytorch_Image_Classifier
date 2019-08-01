import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from torch import optim
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
def LoadData(where  = "./flowers" ):
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = transforms.Compose(
        [transforms.RandomRotation(30), transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip
        (), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(225), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    valication_data = datasets.ImageFolder(valid_dir, transform=test_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainLoader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testLoader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    validationLoader = torch.utils.data.DataLoader(valication_data, batch_size=64, shuffle=True)

    classidx = train_datasets.class_to_idx

    return trainLoader,testLoader,validationLoader,classidx

def SetupNuralNet(ImgNetModel,lr,dropout=0.3):
    if ImgNetModel == 'vgg16':
        model = models.vgg16(pretrained=True)
    #model = models.ImgNetModel(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    #from collections import OrderedDict
    model.classifier = nn.Sequential(nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4096, 1000),
                                     nn.ReLU(), nn.Dropout(dropout), nn.Linear(1000, 250), nn.ReLU()
                                     , nn.Dropout(dropout), nn.Linear(250, 102), nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    if torch.cuda.is_available():
        model.cuda()



    return model,criterion,optimizer


def TrainModel(model,optimizer,criterion,epochs,trainLoader,validationLoader,device='cuda'):
    Train_loss_graph = []
    Test_loss_graph = []
    #epochs = 25
    steps = 0
    running_loss = 0
    print_every = 35
    for epoch in range(epochs):
        for inputs, labels in trainLoader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationLoader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)  ####bueatify result
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        # +++++++++++++++++++
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(validationLoader):.3f}.. "
                      f"Test accuracy: {accuracy / len(validationLoader):.3f}")
                
                Train_loss_graph.append(running_loss)
                Test_loss_graph.append(test_loss)
                running_loss = 0
                model.train()
    return model

def SaveCheckpoint(model,classidx,epchos,ModelName,path):
    model.class_to_idx = classidx
    model.cpu
    checkpoint = {'architecture': ModelName, 'classifier': model.classifier, 'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(), 'epcohs': epchos}

    torch.save(checkpoint, path)