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
train_batches= iter(train_loader)

plt.figure(figsize=(10,12))
X_train, y_train= next(iter(train_batches))
print(X_train.shape, y_train.shape)
print(X_train.size(0))
print(X_train.view(X_train.size(0),-1).shape)
for index in range(100):
    plt.subplot(10, 10, index+1)
    plt.axis('off')
    plt.imshow(X_train[index,:,:,:].numpy().reshape(28,28), cmap='gray')
    plt.title('Class: '+str(y_train[index].item()))
plt.subplots_adjust(hspace=0.5, wspace=0.5) #간격조정
#창을 보인다
plt.show()

#모델을 정의한다
class Model(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()

        self.linear_layer= nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        y=self.linear_layer(x)
        return y


class Layer(nn.Module):
    def __init__(self, input_size, output_size, batch_norm=True, dropout=0.5):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.dropout = dropout

        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),  # LeakyReLU 기본값은 0.01
            self.apply_regularization()
        )

    def apply_regularization(self):
        if self.batch_norm:
            return nn.BatchNorm1d(self.output_size)  # 입력 사이즈를 넣어줘야 함 (입력이 그 앞단의 Linear Layer 의 출력 사이즈가 됨)
        else:
            return nn.Dropout(self.dropout)

    def forward(self, x):
        return self.layer(x)
class DNNModel(nn.Module):
    def __init__(self,input_size, output_size,batch_norm=True,dropout=True):
        super().__init__()
        self.layers= nn.Sequential(
            Layer(input_size, 256, batch_norm, dropout),
            Layer(256, 256, batch_norm, dropout),
            Layer(256, 128, batch_norm, dropout),
            nn.Linear(128, output_size),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.layers(x)

#미니 배치사이즈 128
minibatch_size= 128
input_dim= 28*28
output_dim= 10
model= DNNModel(input_dim, output_dim)
#손실함수를 정의한다
loss_function= nn.NLLLoss()# 로그소프트 맥스 함수이기때문에
#옵티마이저를 정의한다
optimizer= torch.optim.Adam(model.parameters())#아담은 러닝레이트가 필요없다

print(model)
#모델을 트레이닝하는 함수를 정의한다
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


#훈련실행
nb_epochs= 1
progress_interval= 1
early_stop= 1
model,lowest_loss,train_losses,valid_losses= train_model(model,early_stop,nb_epochs,progress_interval)


x_minibaatch, y_minibatch= next(iter(test_loader))
y_test_pred= model(x_minibaatch.view(x_minibaatch.size(0), -1))
pred= torch.argmax(y_test_pred, dim=1)
print(y_test_pred.shape, y_minibatch.shape, pred.shape)
correct= pred.eq(y_minibatch).sum()
print(pred.eq(y_minibatch).sum(),pred.eq(y_minibatch).sum().item())

wrong_idx=pred.ne(y_minibatch).nonzero()[:,0].numpy().tolist()
for index in wrong_idx:
    print(index)


#최종코드

test_loss=0
correct=0
worng_samples, wrong_preds,actual_preds= list(),list(),list()

model.eval()
with torch.no_grad():
    for x_minibatch, y_minibatch in test_loader:
        y_test_pred= model(x_minibatch.view(x_minibatch.size(0), -1))
        test_loss+= loss_function(y_test_pred, y_minibatch).item()
        pred = torch.argmax(y_test_pred, dim=1)
        correct+= pred.eq(y_minibatch).sum().item()

        wrong_idx= pred.ne(y_minibatch).nonzero()[:,0].numpy().tolist()
        for index in wrong_idx:
            worng_samples.append(x_minibatch[index])
            wrong_preds.append(pred[index])
            actual_preds.append(y_minibatch[index])

test_loss/= len(test_loader.dataset)
print('Test set: Average loss: {:.4f}'.format(test_loss))
print('ACCURACY: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


plt.figure(figsize=(12,12))
for index in range(100):
    plt.subplot(10,10,index+1)
    plt.axis('off')
    plt.imshow(worng_samples[index].numpy().reshape(28,28), cmap='gray')
    plt.title('Actual: {}, Pred: {}'.format(actual_preds[index].item(), wrong_preds[index].item()))

#창을 보인다
plt.show()