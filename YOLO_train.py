import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function
from torch.nn import functional as f
import random
from itertools import zip_longest
import torchvision.transforms as transforms
import skimage.io as io
import cv2
from torch import optim
import numpy as np
from YoloCost import Criterion

def ListToTensor(input):
    length = len(input)
    tensor = torch.zeros(length, 5)
    for i, value in enumerate(input):
        words = value.split(" ")
        tensor[i-1,0] = int(words[0])
        tensor[i-1,1] = float(words[1])
        tensor[i-1,2] = float(words[2])
        tensor[i-1,3] = float(words[3])
        tensor[i-1,4] = float(words[4])
    return tensor


# In[3]:

trans = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),])


# In[4]:

# Iterator Images
def grouper(n, iteratable, fillvalue=None):
    args = [iter(iteratable)]*n
    return zip_longest(fillvalue=fillvalue, *args)

# Loading and Reading File
def loadFile(file):
    dataPathes = open(file).read().strip().split('\n')
    return dataPathes

# Reading Grand Truth Labels For Training Images
def loadLabel(year,id):
    labels = open('./VOCdevkit/'+year+'/labels/'+id+'.txt').read().strip().split('\n')
    return labels

# Getting Batch in each epoch
def getBatch(batchAddress, imageSize):
    images = {}
    labels = {}
    for id, data in enumerate(batchAddress):

        if data!=None:
            temp = data.split('/')
            imageID = temp[-1].strip().split('.')[0]
            year = temp[-3]
            images[id] = trans(cv2.resize(io.imread(data), imageSize).transpose((2,0,1))).unsqueeze(0)
            labels[id] = loadLabel(year, imageID)

    return images, labels


# In[5]:

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = (xB - xA ) * (yB - yA)
 
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1])
 
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    return iou


# In[6]:

class YOLONet(nn.Module):
    def __init__(self, extractor):
        super(YOLONet, self).__init__()
        self.features = nn.Sequential(
                # Select Feature
                *list(extractor.children())[:-2]
        )
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(512,1024,3,padding=1)
        self.batchNorm1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024,512,1)
        self.batchNorm2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512,1024,3,padding=1)
        self.batchNorm3 = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(1024,512,1)
        self.batchNorm4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512,1024,3,padding=1)
        self.batchNorm5 = nn.BatchNorm2d(1024)
        self.final = nn.Conv2d(1024,30,1)
        
    def forward(self, input):
        output = self.features(input)
        output = self.maxpool1(output)
        output = f.leaky_relu(self.batchNorm1(self.conv1(output)),0.1)
        output = f.leaky_relu(self.batchNorm2(self.conv2(output)),0.1)
        output = f.leaky_relu(self.batchNorm3(self.conv3(output)),0.1)
        output = f.leaky_relu(self.batchNorm4(self.conv4(output)),0.1)
        output = f.leaky_relu(self.batchNorm5(self.conv5(output)),0.1)
        output = f.dropout(output, p = 0.5)
        output = self.final(output)
        output = f.sigmoid(output)
        return output

S = 7
B = 2
nc = 3
oc = 30
itr = 135
numBacthes = 128
learning_rate = 1e-4
momentum = 0.9
weight_decay = 1e-4
trainImagePath = './train.txt'
train = loadFile(trainImagePath)
numTrain = len(train)
imageSize = (448,448)
cuda = True
landa_coord = 5
landa_nobj = 0.5
print("Resnet18 is loading ... ")
resnet18 = torchvision.models.resnet18(pretrained=True)
print("Resnet was loaded!")

print("Network arch. is crearing")
net = YOLONet(resnet18)
for param in net.features.parameters():
    param.requires_grad = False
print("Network was Created")

input = V(torch.randn(1,nc,imageSize[0], imageSize[1]))
parameters = (p for p in list(net.parameters())[-12:])
#optimizer = optim.SGD(params = parameters, lr = learning_rate, momentum=momentum, weight_decay = weight_decay)
optimizer = optim.Adam(params = parameters, lr = learning_rate)
outf = './snapshot'

# item = train[2:3]
# print("Getting one image and its label ...")
# cpu_images, cpu_labels = getBatch(item, imageSize)
# print("Image was fetched. ")

# optimizer.zero_grad()
# input.data.resize_(cpu_images[0].size()).copy_(cpu_images[0]);
# output = net(input)

print("Creating Loss Function")
criterion = Criterion(S, B, landa_coord, landa_nobj)
print("Loss Function was created!")

# print("Target is converting to tensor")
# target = V(ListToTensor(cpu_labels[0]))
# print(target.data)
# print("Target was converted")
# instanceCost = criterion(output, target)
# print("InstanceCost = ", instanceCost)

# print("Running backprop on instanceCost: ")
# instanceCost.backward()

print("Start Training ....")

if cuda:
    net.cuda()
    input = input.cuda()

for epoch in range(itr):
    random.shuffle(train)
    currentBatch = 1
    for items in grouper(numBacthes, train):
        cpu_images, cpu_labels = getBatch(items, imageSize)
        loss = 0
        optimizer.zero_grad()
        for imageIndex in range(len(cpu_images)):
            input.data.resize_(cpu_images[imageIndex].size()).copy_(cpu_images[imageIndex]);
            output = net(input)
            output = output.cpu()
            target = V(ListToTensor(cpu_labels[imageIndex]))
            cost = criterion(output, target)
            loss += cost.data[0]
            cost.backward(retain_variables=True)
        optimizer.step()
        if currentBatch % 20             == 0:
            print("(%d,%d) -> Current Batch Loss:%f"%(epoch+1,currentBatch,loss/numBacthes))
        currentBatch += 1
    torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (outf, epoch))
print("Finish Training ....")