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
import math
import time

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

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classesColor = [(0,0,0),
                (0,0,255),
                (0,255,0),
                (0,255,255),
                (255,0,0),
                (255,0,255),
                (255,255,0),
                (255,255,255),
                (0,50,100),
                (0,100,50),
                (100,50,0),
                (100,0,50),
                (50,0,100),
                (50,100,0),
                (100,150,255),
                (100,150,255),
                (150,100,255),
                (150,255,100),
                (255,100,150),
                (255,150,100)]
floor = math.floor
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
def imshow(img, output_label, threshold, B, S):
    numpyImg = cv2.resize(img, (448,448))
    size = numpyImg.shape
    W = size[0]
    H = size[1]
    numpyOut = output_label.squeeze(0).numpy()            
    BestClass = np.zeros((S,S))
    BestClass = np.argmax(numpyOut[10:,:,:], axis=0)
    XBBox = np.stack((numpyOut[0 + 0,:,:],numpyOut[0 + 5,:,:]))*W
    YBBox = np.stack((numpyOut[1 + 0,:,:],numpyOut[1 + 5,:,:]))*H
    WBBox = np.stack((numpyOut[2 + 0,:,:],numpyOut[2 + 5,:,:]))*W
    HBBox = np.stack((numpyOut[3 + 0,:,:],numpyOut[3 + 5,:,:]))*H
    P = np.stack((numpyOut[4,:,:],numpyOut[9,:,:]), axis=0)
    bestPindex = np.argmax(P, axis=0)
    bestP = np.max(P,axis=0)
    bx, by = np.where(bestP>=threshold)
    if bx.size > 0:
        ix1, iy1 = np.where(bestPindex == 0)
        ix2, iy2 = np.where(bestPindex != 0)
        indexT = np.zeros((2,7,7))
        indexT[0,ix1, iy1] = 1
        indexT[0,ix2, iy2] = 0
        indexT[1,ix1,iy1] = 0
        indexT[1,ix2,iy2] = 1
        XBBOX_final = np.sum(XBBox * indexT,axis=0)
        YBBOX_final = np.sum(YBBox * indexT,axis=0)
        WBBOX_final = np.sum(WBBox * indexT,axis=0)
        HBBOX_final = np.sum(HBBox * indexT,axis=0)
        pt1 = np.stack((np.floor(XBBOX_final-(WBBOX_final/2)),np.floor(YBBOX_final-(HBBOX_final/2))))
        pt2 = np.stack((np.floor(XBBOX_final+(WBBOX_final/2)),np.floor(YBBOX_final+(HBBOX_final/2))))
        pt1[pt1<0] = 0
        pt1[pt1>448] = 448
        pt2[pt2<0] = 0
        pt2[pt2>448] = 448
        for i in range(bx.size):
            print("boxX=%d, boxY=%d, x=%f ,y=%f ,w=%f, h=%f, p(object)=%f" %(bx[i], by[i], XBBOX_final[bx[i],by[i]], YBBOX_final[bx[i],by[i]], WBBOX_final[bx[i],by[i]], HBBOX_final[bx[i],by[i]], bestP[bx[i],by[i]]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(numpyImg,classes[int(BestClass[bx[i],by[i]])],(int(pt1[0,bx[i],by[i]]),int(pt1[1,bx[i],by[i]])), font, .5,classesColor[int(BestClass[bx[i],by[i]])],2,cv2.LINE_AA)
            cv2.rectangle(numpyImg, (int(pt1[0,bx[i],by[i]]),int(pt1[1,bx[i],by[i]])),
                         (int(pt2[0,bx[i],by[i]]),int(pt2[1,bx[i],by[i]])),classesColor[int(BestClass[bx[i],by[i]])],int(bestP[bx[i],by[i]]*5))

    return numpyImg
    

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
numBacthes = 64
learning_rate = 1e-4
momentum = .9
weight_decay = 1e-3
trainImagePath = './2010_train.txt'
train = loadFile(trainImagePath)
numTrain = len(train)
print(numTrain)
imageSize = (448,448)
cuda = True
landa_coord = 5
landa_nobj = .5
threshold = 0.3
print("Resnet18 is loading ... ")
resnet18 = torchvision.models.resnet18(pretrained=True)
print("Resnet was loaded!")

print("Loading Network: ....")
net = YOLONet(resnet18)
net.load_state_dict(torch.load('./snapshot5_final.pth'))
net.eval()
print("Network was loaded")

input = V(torch.randn(1,nc,imageSize[0], imageSize[1]))

#capture from camera at location 0
cap = cv2.VideoCapture(0)
#set the width and height, and UNSUCCESSFULLY set the exposure time
cap.set(3,448)
cap.set(4,448)
time.sleep(2)

cv2.namedWindow("MainWindows")
cv2.moveWindow("MainWindows", 448,448)
while True:
    ret, img = cap.read()
    inp = trans(cv2.resize(img, imageSize).transpose((2,0,1))).unsqueeze(0)
    input.data.resize_(inp.size()).copy_(inp)
    output = net(input)
    numpyImg = imshow(img, output.data, threshold, B, S)
    cv2.imshow("MainWindows", numpyImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break            

cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()