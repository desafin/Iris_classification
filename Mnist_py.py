import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#데이터를 토치비전에서 다운받는다
train_rawdata= datasets.MNIST(root='mnist_data', train=True, download=True, transform=transforms.ToTensor())
#테스트 데이터 셋을 다운
test_datase= datasets.MNIST(root='mnist_data', train=False, download=True, transform=transforms.ToTensor())

#트레이닝 데이터셋의 갯수를 출력하기
print('Number of training images: ', len(train_rawdata))
#테스트 데이터셋의 갯수를 출력하기
print('Number of test images: ', len(test_datase))



VALIDATION_RATE= 0.2
train_indices, val_indices,_,_= train_test_split(range(len(train_rawdata)),#인덱스 번호
                                                 train_rawdata.targets,stratify=train_rawdata.targets, #균등분포
                                                 test_size=VALIDATION_RATE
                                                 )

#torch.utils.data.Subset을 이용하여 트레이닝 데이터셋과 벨리데이션 데이터셋을 나눈다
train_dataset= Subset(train_rawdata, train_indices)
val_dataset= Subset(train_rawdata, val_indices)

#데이터 사이즈를 확인
print('Number of training images: ', len(train_dataset))
print('Number of validation images: ', len(val_dataset))
print('Number of test images: ', len(test_datase))

#미니 배치를 만든다
BATCH_SIZE= 128
train_loader= DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader= DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader= DataLoader(test_datase, batch_size=BATCH_SIZE, shuffle=False)

#미니 배치 하나만 가져와서 이미지 시각화
train_batch= iter(train_loader)

plt.figure(figsize=(10,12))
X_train, y_train= next(iter(train_batch))
print(X_train.shape, y_train.shape)
print(X_train.size(0))
print(X_train.view(X_train.size(0),-1).shape)
for index in range(100):
    plt.subplot(10, 10, index+1)
    plt.axis('off')
    plt.imshow(X_train[index,:,:,:].numpy().reshape(28,28), cmap='gray')
    plt.title('Class: '+str(y_train[index].item()))

#창을 보인다
plt.show()

#모델을 정의한다
class Model(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()

        self.linear_layer= nn.Squential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        y=self.linear_layer(x)
        return y

#미니 배치사이즈 128
minibatch_size= 128
input_dim= 28*28
output_dim= 10
model= Model(input_dim, output_dim)
#손실함수를 정의한다
loss_function= nn.NLLLoss()# 로그소프트 맥스 함수이기때문에
#옵티마이저를 정의한다
optimizer= torch.optim.Adam(model.parameters())#아담은 러닝레이트가 필요없다


