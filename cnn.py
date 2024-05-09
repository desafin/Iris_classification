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


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers= nn.Sequential(
            nn.Conv2d(1, 32,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.linear_layers= nn.Sequential(
            nn.Linear(128*3*3, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )
        def forward(self, x):
            x= self.conv_layers(x)
            x= x.view(x.size(0), -1)
            x= self.linear_layers(x)
            return x


model= CNNModel()
print(model)



#손실함수를 정의한다
loss_function= nn.NLLLoss()# 로그소프트 맥스 함수이기때문에
#옵티마이저를 정의한다
optimizer= torch.optim.Adam(model.parameters())#아담은 러닝레이트가 필요없다



def train_model(model,early_stop,n_epochs,progress_interval):
    train_losses,vaild_losses,lowest_loss= list(),list(),np.inf
    lowest_epoch=0
    for epoch in range(n_epochs):
        train_loss,valid_loss= 0,0
        #모델 훈련
        model.train()
        for x_minibatch, y_minibatch in train_batches:
            y_minibatch_pred= model(x_minibatch.view(x_minibatch.size(0), -1))
            loss= loss_function(y_minibatch_pred, y_minibatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+= loss.item()
        train_loss= train_loss/len(train_batches)
        train_losses.append(train_loss)

        #모델 검증
        model.eval()
        with (torch.no_grad()):
            for x_minibatch, y_minibatch in val_loader:
                y_minibatch_pred= model(x_minibatch.view(x_minibatch.size(0), -1))
                loss= loss_function(y_minibatch_pred, y_minibatch)
                valid_loss+= loss.item()

        vaild_loss=valid_loss/len(val_loader)
        vaild_losses.append(vaild_loss)

        if vaild_losses[-1]<lowest_loss:
            lowest_loss=vaild_losses[-1]
            lowest_epoch=epoch
            best_model=deepcopy(model.state_dict())
        else:
            if early_stop > 0 and lowest_epoch + early_stop<epoch:
                print('Early stopping at epoch', epoch)
                break

        if (epoch % progress_interval)==0:
            print(train_losses[-1],vaild_losses[-1],lowest_loss,lowest_epoch,epoch)

    model.load_state_dict(best_model)
    return model,lowest_loss,train_losses,vaild_losses


nb_epochs= 100
progress_interval= 3
early_stop= 30
model,lowest_loss,train_losses,valid_losses= train_model(model,early_stop,nb_epochs,progress_interval)

