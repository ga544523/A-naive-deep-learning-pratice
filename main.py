
import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import pandas as pd
import csv
import gc
'''
paths='/home/ipx/data/append'
cons=4600
def batch_rename(path):
    for fname in os.listdir(path):
        new_fname=fname.split('.')
        print(new_fname)
        os.rename(os.path.join(path, fname), os.path.join(path, str( (int)(new_fname[0])-1151+cons) )+'.png' )

def batch_renamelabel(path):
    for fname in os.listdir(path):
        new_fname=fname.split('_')
        print(new_fname)
        os.rename(os.path.join(path, fname), os.path.join(path, str( (int)(new_fname[0])-1151+cons) )+'_'+new_fname[1] )


#batch_renamelabel(paths)
#print(1/0)


f = open('/home/ipx/data/totallabel/Labels.txt', 'r')
lines=f.readlines() # 讀取檔案內容的每一行文字為陣列
for line in lines:
    x = line.split()
    print(x)
    with open('output.csv', 'a+', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料
        writer.writerow([ str(int(x[0])-1151+cons )+'.png', x[1] ])
f.close() # 關閉檔案
'''

import torch.nn.functional as F
import pandas as pd
from random import shuffle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import cmath
import math
import torchvision.models as models
from torchvision.utils import save_image, make_grid
import random
import threading
import heapq as hq
import time
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F



device1 = torch.device("cuda:0")

directory = '/home/ipx/venv/'
df = pd.read_csv(directory + "output.csv")
data=[]

print(torch.cuda.is_available())
print(torch.__version__)
'''
with open('/home/ipx/venv/output.csv', newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)

  # 寫入一列資料
  with open('/home/ipx/venv/output1.csv', 'a+', newline='') as csvfile1:
      writer1 = csv.writer(csvfile1)
      # 以迴圈輸出每一列
      cnt=0
      cla=0
      cona=575
      conb=1150
      for row in rows:
        data.append(row)
        print(row)
        writer1.writerow([row[0], row[1],cla])
        cnt=cnt+1
        cla+=int(cnt/cona)
        cnt%=cona

        if(cla==6):
            cona=1150
'''
with open('/home/ipx/venv/output.csv', newline='') as csvfile:
  rows = csv.reader(csvfile)
  for row in rows:
    data.append(row)




shuffle(data)
imgsize=200

totalimagedir='/home/ipx/data/totalimage/'
totallabeldir='/home/ipx/data/totallabel/'

batchsize=32



# 引入 time 模組
import time

class segment(nn.Module):

    def __init__(self, num_classes=200 * 200):
        super(segment, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(1, 128, kernel_size=6, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(128, 128, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 128, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(128, 2, kernel_size=1),

        )

        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=10, mode='bilinear')
        )
    def forward(self, x):
        num=x.size(0)
        if(test_mode==True):
            num=1
            x = self.features(x)
            '''
            out = torch.zeros([num, 2, 20, 20], dtype=torch.float64, device=torch.device('cuda:0'))
            for t in range(0, num , 1):
                for i in range(0, 20, 1):
                    for j in range(0, 20, 1):
                        out[t][0][i][j] = x[t][0][i][j]
                        out[t][1][i][j] = x[t][1][i][j]
                        print(out[t][0][i][j],out[t][1][i][j])
            out = out.view(num , 2, 20, 20)
            #out = self.upscale(out)
            '''
            m = torch.nn.Softmax(dim=1)
            out = m(x)
            '''
            for t in range(0, num , 1):
                for i in range(0, 20, 1):
                    for j in range(0, 20, 1):
                        print(out[t][0][i][j],out[t][1][i][j])
            #print(out.size())
            '''

        else:
            x=self.features(x)
            #print(x.size())
            '''
            out=torch.zeros([num,2,20, 20], dtype=torch.float64, device=torch.device('cuda:0'))
            for t in range(0,num,1):
                for i in range(0,20,1):
                    for j in range(0,20,1):
                        out[t][0][i][j]=x[t][0][i][j]
                        out[t][1][i][j] = x[t][1][i][j]



            out = out.view(num, 2, 20, 20)
            '''
            #out=self.upscale(out)
            m=torch.nn.Softmax(dim=1)
            out=m(x)
        print(out.size())
        return out

class classification(nn.Module):
    def __init__(self, num_classes=200 * 200):
        super(classification, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(1, 128, kernel_size=6, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 64, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 20, kernel_size=1),
            nn.Conv2d(20, 8, kernel_size=36),
        )

    def forward(self, x):
        num=x.size(0)
        if(test_mode==True):
            num=1
            x = self.features(x)
            out = torch.zeros([num, 2, 20, 20], dtype=torch.float64, device=torch.device('cuda:0'))
            for t in range(0, num , 1):
                for i in range(0, 20, 1):
                    for j in range(0, 20, 1):
                        out[t][0][i][j] = x[t][0][i][j]
                        out[t][1][i][j] = x[t][1][i][j]
                        print(out[t][0][i][j],out[t][1][i][j])
            out = out.view(num , 2, 20, 20)
            #out = self.upscale(out)
            m = torch.nn.Softmax(dim=1)
            out = m(out)
            for t in range(0, num , 1):
                for i in range(0, 20, 1):
                    for j in range(0, 20, 1):
                        print(out[t][0][i][j],out[t][1][i][j])
            #print(out.size())
        else:
            x=self.features(x)
            #print(x.size())
            m=torch.nn.Softmax(dim=1)
            out=m(x)
        print(out.size())
        return out

classification_net=classification()

trainmode=5


segment_net = segment()


if(trainmode==0):
    classification_net=classification_net.to(device1)
else:
    segment_net = segment_net.to(device1)


def testing():
    global batchsize
    batchsize=1
    if(trainmode==0):
        classification_net = torch.load('/home/ipx/venv/model_class1.pkl')
        classification_net.eval()
        classification_net = classification_net.to(device1)
        for i in range(0, len(data), 1):
            if (data[i][1] == '1'):
                s = data[i][0]

                img = cv2.imread(totalimagedir + s, cv2.IMREAD_GRAYSCALE)
                imgshow = cv2.imread(totalimagedir + s)
                img = cv2.resize(img, dsize=(imgsize, imgsize))

                imgshow = cv2.resize(imgshow, dsize=(imgsize, imgsize))

                img = img.astype(np.float32)

                img = torch.from_numpy(img)
                img = img.view(1, 1, imgsize, imgsize)

                output = classification_net(img.to(device1)).to(device1)
                cv2.imshow('orginion', imgshow)
                maxr=0
                id=0
                for i1 in range(0, 8, 1):
                    if(output[0][i1]>maxr):
                        maxr=output[0][i1]
                        id=i1

                print('ans_label=',id)
                cv2.waitKey(0)
    else:
        segment_net = torch.load('/home/ipx/venv/segment_class'+str(trainmode)+'.pkl')
        segment_net.eval()
        segment_net = segment_net.to(device1)


        for i in range(0,len(data),1):
            if(data[i][1]=='1' and int(data[i][2])-int('0')==int(trainmode-1) ):
                s=data[i][0]

                img=cv2.imread(totalimagedir+s, cv2.IMREAD_GRAYSCALE)
                imgshow=cv2.imread(totalimagedir+s)
                img = cv2.resize(img, dsize=(imgsize, imgsize))

                imgshow = cv2.resize(imgshow, dsize=(imgsize, imgsize))

                img = img.astype(np.float32)

                img=torch.from_numpy(img)
                img = img.view(1, 1, imgsize, imgsize)

                output=segment_net(img.to(device1)).to(device1)
                cv2.imshow('orginion', imgshow)

                for i1 in range(0,10,1):
                    for j1 in range(0,10,1):
                        if(output[0][0][i1][j1]>output[0][1][i1][j1]):
                            for i2 in range(0, 20, 1):
                                for j2 in range(0, 20, 1):
                                    imgshow[i1*20+i2][j1*20+j2][0] = 255

                cv2.imshow('result', imgshow)

                label=cv2.imread(totallabeldir + str.split(s,'.')[0]+'_label.PNG', cv2.IMREAD_GRAYSCALE)
                label = cv2.resize(label, dsize=(imgsize, imgsize))
                cv2.imshow('abel', label)

                cv2.waitKey(0)

test_mode=False
if(test_mode==True):
    testing()

#binary_net=binary().to(device1)


segment_optimizer = torch.optim.SGD(segment_net.parameters(), lr=0.0001, momentum=0.9)

class_optimizer = torch.optim.SGD(classification_net.parameters(), lr=0.0001, momentum=0.9)


drawx=[]
drawy=[]

count_update=0
def custom_segement_loss(output,label):
    global count_update
    global drawy
    global drawx
    defect_label_image=0
    defect_label_image_defect_free=0
    flag=0
    half=int(len(label))

    for i in range(0,half,1):
        s = label[i]

        img = cv2.imread(totallabeldir + str.split(s,'.')[0]+'_label.PNG', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(imgsize, imgsize))
        img = img.astype(np.float32)
        for i1 in range(0,imgsize,1):
            for j1 in range(0,imgsize,1):
                if(img[i1][j1]>0):
                    img[i1][j1]/=255
        tensorimg=torch.from_numpy(img)
        del img
        tensorimg = tensorimg.view(1,1,imgsize,imgsize)
        if(flag==0):
            defect_label_image=tensorimg
            flag=1
        else:
            defect_label_image = torch.cat((defect_label_image, tensorimg), 0)
    '''
    defect_label_image=defect_label_image.view(batchsize,1,512*512)
    print(defect_label_image.size())
    print(1/0)
    '''
    sum=0
    now = time.time()
    cnt=0
    for b in range(0,batchsize,1):
        for h in range(0,10,1):
            for w in range(0,10,1):
                defectarea=0
                for h1 in range(0,20,1):
                    for w1 in range(0,20,1):
                        if (defect_label_image[b][0][h*20+h1][w*20+w1] > 0.5):
                            defectarea+=1
                if(defectarea>70):
                   sum+=-torch.log(output[b][0][h][w])*0.8
                else:
                    sum += -torch.log(output[b][1][h][w])*0.2
    sum/=(10*10)
    sum/=batchsize
    print('segment_loss=', sum)
    drawx.append(count_update)
    drawy.append(sum.item())
    count_update += 1
    if(count_update%50==0):
        plt.plot(drawx, drawy)
        plt.savefig('loss'+str(count_update)+'.png')
    now1 = time.time()
    #print('loss time',now1-now)
    return sum

def custom_class_loss(output, imageclass):
    global count_update
    sum = 0
    output=output.view(batchsize,8)

    for b in range(0, batchsize, 1):
        for h in range(0, 8, 1):
            sum += -torch.log(output[b][h])*imageclass[b][h]
    sum /= 8
    sum /= batchsize

    print('class_loss=', sum)
    drawx.append(count_update)
    drawy.append(sum.item())
    count_update += 1
    if (count_update % 50 == 0):
        plt.plot(drawx, drawy)
        plt.savefig('loss' + str(count_update) + '.png')

    return sum


def merge(defect_data,defect_free_data,defect_label,defete_free_labal):

    totalimage_tmp=traindata_defect
    totallabel_tmp=defect_label
    return totalimage_tmp,totallabel_tmp


def train_segment(totalimage,totallabel):


    segment_optimizer.zero_grad()

    train_totallabel = np.array(totallabel)
    imagename = train_totallabel[:, 1]
    train_totalimage = totalimage.to(device1)

    output = segment_net(train_totalimage.to(device1)).to(device1)

    segment_loss = custom_segement_loss(output, imagename)

    now = time.time()
    segment_loss.backward()
    now1 = time.time()
    #print('lossbacktime', now1 - now)
    return

def train_class(totalimage,totallabel):
    class_optimizer.zero_grad()
    totallabel=np.array(totallabel)
    imageclass_tmp = totallabel[:,2]

    imageclass_tmp = imageclass_tmp.astype(np.float32)
    imageclass = torch.from_numpy(imageclass_tmp)
    flag=0

    del imageclass_tmp

    imageclass = imageclass.to(device1)
    output = classification_net(totalimage.to(device1))

    image_label=0
    for i in range(0,batchsize,1):
        one_hot=torch.zeros([8], dtype=torch.float64)
        one_hot[int(imageclass[i])]=1
        one_hot=one_hot.view(1,8)
        if(flag==0):
            flag+=1

            image_label=one_hot

        else:

            image_label=torch.cat( (image_label,one_hot),0)
    class_loss = custom_class_loss(output, image_label)
    class_loss.backward()

    print('class_loss=',class_loss)
    return

for t in range(0,1000000,1):
    shuffle(data)

    print('epoch ',t)
    positive=[]
    negative=[]

#    torch.save(binary_net,'model_binary,pkl')

    for i in range(0,len(data),1):
        if(data[i][1]=='0' or int(data[i][2])-int('0')!=int(trainmode-1) ):
            1
            #negative.append(data[i][0])
        else:
            positive.append([data[i][0],data[i][2] ])

    print(len(positive))

    #get defect
    for j in range(0,len(positive)-batchsize,batchsize):
        traindata_defect = 0
        trainlabel_defect = []
        flag = 0

        for k in range(0,batchsize,1):
            s=positive[j+k][0]

            img=cv2.imread(totalimagedir+s, cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,dsize=(imgsize, imgsize))

            img = img.astype(np.float32)

            tensorimg=torch.from_numpy(img)
            del img
            tensorimg = tensorimg.view(1,1,imgsize,imgsize)
            #img=tf.image.rgb_to_grayscale(img)
            trainlabel_defect.append([1.0,s,positive[j+k][1] ])

            if(flag==0):
                traindata_defect=tensorimg
                flag=1
            else:
                traindata_defect=torch.cat( (traindata_defect,tensorimg),0)

        traindata_defect_free=0
        trainlabel_defect_free=[]

        # get defect-free
        '''
        for j in range(0, batchsize, 1):
    
            s = negative[j]
    
            img = cv2.imread(totalimagedir + s, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32)
    
            img = torch.from_numpy(img)
            img = img.view(1, 1, 512, 512)
            # img=tf.image.rgb_to_grayscale(img)
            trainlabel_defect_free.append([0, s])
    
            if (flag == 0):
                traindata_defect_free = img
                flag = 1
            else:
                traindata_defect_free = torch.cat((traindata_defect_free, img), 0)
        '''

        #merge
        totalimage,totallabel=merge(traindata_defect,traindata_defect_free,trainlabel_defect,trainlabel_defect_free)
        now = time.time()
        #segment loss

        if(trainmode==0):
            train_class(totalimage, totallabel)
            class_optimizer.step()
        else:
            train_segment(totalimage,totallabel)
            segment_optimizer.step()

        now1 = time.time()
        #print('train total',now1-now)
        #binary loss

        '''
        copyoutput=output
        tensor_binary = torch.zeros([batchsize*2,512], dtype=torch.float32, device=torch.device('cuda:0'))
        for t in range(0,batchsize*2,1):
            for i in range(0,512,1):
                tensor_binary[t][i]=-9999999
                for j in range(0,512,1):
                    if(copyoutput[t][0][i][j]>tensor_binary[t][i]):
                        tensor_binary[t][i]=copyoutput[0][0][i][j]
        '''
        totalimage=0
        totallabel=0
        if(trainmode==0):
            torch.save(classification_net, 'model_class1.pkl')
        else:
            torch.save(segment_net, 'segment_class'+str(trainmode)+'.pkl')
    positive.clear()
    gc.collect()