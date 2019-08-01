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
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
data_dir = "./flowers"
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
import matplotlib.pyplot as plt
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

# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint_cpu():
    checkpoint = torch.load('checkpoint.pth')
   
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters(): param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    #model = model
    device = torch.device('cpu')
    model.to(device)
    #torch.save(model, '/home/workspace/ImageClassifier/temp')
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pilImg = Image.open(f'{image}' + '.jpg')
    TransformImgFeeder = transforms.Compose([transforms.Resize(226),transforms.CenterCrop(224),transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    pic_tensor = TransformImgFeeder(pilImg)
    numpyArray = np.array(pic_tensor)
    return numpyArray
def predict(image_path, model,top_k = 1):
    model.eval()
    image = process_image(image_path)
    #pytorchtensor = torch.tensor(image)
    
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    
    pytorch_tensor = image_tensor.unsqueeze(0) #add a 1 as the first argument of our tensor.
    output = model.forward(pytorch_tensor)
    output1 = torch.exp(output) #the predicted probability
    prob,indices = output1.topk(top_k)
    top_probs = prob.detach().numpy().tolist()[0]
    top_labs = indices.detach().numpy().tolist()[0]
    #[dict[model.class_to_idx] for model.class_to_idx in indices ]
    #map(model.class_to_idx.get, indices)
    lis=[]
    #lis = Null
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    #print(idx_to_class)
    for i in top_labs:
        #val = str(i)
        lis.append(idx_to_class[i])
    #print("list values = ",lis)
    return top_probs,lis

def PredictProb(imagePath,model,top_k):
    ##lets get the image 
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    imageConv = process_image(imagePath)
    #imshow(imageConv, ax=None, title=None)
    topP, topC = predict(imagePath, model,top_k = int(top_k))
    
    lis1 =[]
    for i in topC:
        val = str(i)
        lis1.append(cat_to_name[val])
    print(lis1, topP)
    
   
    #return lis1,topP
    
    
    
    
        
    # TODO: Process a PIL image for use in a PyTorch model
#model = load_checkpoint_cpu()
#criterion = nn.NLLLoss()
#optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)

#train_dir = data_dir + '/train'
#model =load_checkpoint_cpu()

#print("model prob predictions ===============",model)

model =load_checkpoint_cpu()
image_graph = test_dir+'/10/'+'image_07090'
#img_test_o = process_image(image_graph)
#img_test_o = process_image(image_graph)    
#PredictProb(image_graph,model)

#print(flowerc,prob)
#trainloader, v_loader, testloader ,classidx= LoadData('./flowers/')
#SaveCheckpoint(model,classidx,epchos='20',ModelName='vgg19')


#print(prob,classes)
