import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# sklearn에서 iris 데이터셋을 불러옵니다.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# 데이터 로드
iris = load_iris()
X = iris['data']  # 데이터 포인트
y = iris['target']  # 레이블
names = iris['target_names']  # 타겟 이름 (클래스 이름)
feature_names = iris['feature_names']  # 피처 이름

# 데이터를 학습용과 테스트용으로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
std_scaler = StandardScaler()
std_scaler.fit(X_train)  # 학습 데이터를 기준으로 표준화 파라미터를 계산
X_train_tensor = torch.from_numpy(std_scaler.transform(X_train)).float()
X_test_tensor = torch.from_numpy(std_scaler.transform(X_test)).float()
y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

# 텐서의 차원을 출력합니다.
print(X_train_tensor.shape)
print(y_train_tensor.shape)
print(X_test_tensor.shape)
print(y_test_tensor.shape)

# 하이퍼파라미터 설정
nb_epochs = 1000000  # 학습 에포크 수
minibatch_size = 120  # 미니배치 크기

# 신경망 모델 정의
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, 120),
            nn.LeakyReLU(0.1),
            nn.Linear(120, 110),
            nn.LeakyReLU(0.1),
            nn.Linear(110, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 90),
            nn.LeakyReLU(0.1),
            nn.Linear(90, 80),
            nn.LeakyReLU(0.1),
            nn.Linear(80, 70),
            nn.LeakyReLU(0.1),
            nn.Linear(70, 60),
            nn.LeakyReLU(0.1),
            nn.Linear(60, 50),
            nn.LeakyReLU(0.1),
            nn.Linear(50, 40),
            nn.LeakyReLU(0.1),
            nn.Linear(40, 30),
            nn.LeakyReLU(0.1),
            nn.Linear(30, 20),
            nn.LeakyReLU(0.1),
            nn.Linear(20, 10),
            nn.LeakyReLU(0.1),
            nn.Linear(10, 5),
            nn.LeakyReLU(0.1),
            nn.Linear(5, output_dim),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.linear_layers(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

# 모델, 손실 함수, 최적화 알고리즘 초기화
input_dim = X_train_tensor.shape[1]  # 입력 차원
output_dim = 3  # 출력 차원 (클래스 수)
model = Model(input_dim, output_dim)
criterion = nn.NLLLoss()  # NLLLoss는 LogSoftmax의 결과를 입력으로 받습니다.
optimizer = torch.optim.Adam(model.parameters())  # Adam 최적화 알고리즘

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# 모델 초기화
model.init_weights()

# 학습 루프
for index in range(nb_epochs):
    indices = torch.randperm(X_train_tensor.size(0), device=device)
    x_batch_list = X_train_tensor.index_select(0, indices).split(minibatch_size)
    y_batch_list = y_train_tensor.index_select(0, indices).split(minibatch_size)

    break_flag = False  # 플래그 초기화

    for x_minibatch, y_minibatch in zip(x_batch_list, y_batch_list):
        y_minibatch_pred = model(x_minibatch)
        loss = criterion(y_minibatch_pred, y_minibatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 손실이 0.1보다 작으면 학습 중단
        if loss.item() < 0.05:
            print(f'Early stopping at epoch: {index} | Loss: {loss.item()}')
            break_flag = True  # 플래그 설정
            break

    if break_flag:  # 플래그가 설정되면 에포크 루프도 중단
        break

    if index % 100 == 0:
        print(f'Epoch: {index} | Loss: {loss.item()}')
# # 모델 평가
# model.eval()
# with torch.no_grad():
#     y_pred_list = []
#     for x_test_batch in X_test_tensor.split(minibatch_size):
#         y_test_pred = model(x_test_batch)
#         y_pred_list.extend(torch.argmax(y_test_pred, dim=1).tolist())
#
# # 예측 결과 출력
# y_pred_tensor = torch.tensor(y_pred_list)
# print(y_pred_tensor.shape)


# 모델 평가 및 예측 결과 계산
model.eval()
y_pred_list = []
X_test_processed = std_scaler.transform(X_test)  # 실제 피처 데이터 표준화

with torch.no_grad():
    for x_test_batch in torch.tensor(X_test_processed, device=device).float().split(minibatch_size):
        y_test_pred = model(x_test_batch)
        y_pred_list.extend(torch.argmax(y_test_pred, dim=1).tolist())

# 예측 결과 시각화
fig, ax = plt.subplots()
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_list, cmap='viridis', label='Predicted')
legend1 = ax.legend(*scatter.legend_elements(), title="Predictions")
ax.add_artist(legend1)
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='cool', marker='x', label='Actual')
legend2 = ax.legend(*scatter.legend_elements(), title="Actual")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Iris Classification Predictions vs Actual')
plt.show()

# 모델 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x_test_batch, y_test_batch in zip(X_test_tensor.split(minibatch_size), y_test_tensor.split(minibatch_size)):
        y_test_pred = model(x_test_batch)
        _, predicted = torch.max(y_test_pred.data, 1)
        total += y_test_batch.size(0)
        correct += (predicted == y_test_batch).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test set: {accuracy}%')
