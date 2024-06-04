import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
from utils import *
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

valid_data_path = './beans/valid_seperated/'

validation_loader = load_data_validation(valid_data_path)


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
model.load_state_dict(torch.load("alexnet_baseline2.pth"))

for X_batch, y_batch in validation_loader:
    y_pred = model(X_batch)
    pred_label = torch.argmax(y_pred, axis=1)
    cf_matrix = confusion_matrix(pred_label, y_batch)
    break

labels = ['Healthy', 'Unhealthy']

sns.heatmap(cf_matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)

# add labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.savefig("Beans_Classification_Confusion_Matrix.png")