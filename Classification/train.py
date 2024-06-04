import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
from utils import *
import torch.nn as nn
import torch.optim as optim

train_data_path = './beans/train_seperated/'
valid_data_path = './beans/valid_seperated/'

train_loader = load_data(train_data_path)
validation_loader = load_data(valid_data_path)

train_loss = []
valid_loss = []

train_acc = []
valid_acc = []


class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3,96,11,4,0),
            nn.MaxPool2d(3,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,256,5,1,2),
            nn.MaxPool2d(3,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,2,0),
            nn.ReLU(inplace=True),
        )

        self.ann = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(6*6*256,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

        self.classifer = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.head(x)
        h = x.view(x.shape[0], -1)
        x = self.ann(h)
        x = self.classifer(x)
        return x



model = AlexNet(2)
#model.load_state_dict(torch.load("alexnet1.pth"))
print(model)

### train the model

def train(model, epoch, criterion, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    total_loss = 0
    batch_count = 0
    total_acc = 0
    for X_batch, y_batch in train_loader:
        batch_count+=1
        y_pred = model(X_batch)
        acc = get_accuracy(y_pred, y_batch)
        total_acc+=acc
        loss = criterion(y_pred, y_batch)
        total_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("Running loss:" + str(loss))
    train_loss.append(total_loss/batch_count)
    train_acc.append(total_acc/batch_count)
    print("Training loss: " + str(total_loss/batch_count) + "   Training accuracy: " + str((total_acc/batch_count)*100))



def evaluate(model, epoch, criterion):
    model.eval()
    batch_count = 0
    total_loss = 0
    total_acc = 0
    for X_batch, y_batch in validation_loader:
        batch_count+=1
        y_pred = model(X_batch)
        acc = get_accuracy(y_pred, y_batch)
        total_acc+=acc
        loss = criterion(y_pred, y_batch)
        total_loss+=loss
    valid_loss.append(total_loss/batch_count)
    valid_acc.append(total_acc/batch_count)
    print("Validation loss: " + str(total_loss/batch_count)+ "   Validation accuracy: " + str((total_acc/batch_count)*100))


n_epochs = 50
lr = .01
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

for epoch in range(n_epochs):
    train(model,epoch,criterion,lr)
    evaluate(model,epoch,criterion)


torch.save(model.state_dict(), "alexnet_baseline2.pth")



train_loss = [elem.tolist() for elem in train_loss]
valid_loss = [elem.tolist() for elem in valid_loss]
train_acc = [elem.tolist() for elem in train_acc]
valid_acc = [elem.tolist() for elem in valid_acc]

create_plots(train_loss, valid_loss, n_epochs, "losses")
create_plots(train_acc, valid_acc, n_epochs, "accuracy")














