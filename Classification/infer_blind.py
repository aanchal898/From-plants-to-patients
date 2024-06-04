import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
from utils import *
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import pandas as pd


test_data_path = './beans/test/'


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

imgs = []

imgs_paths = glob.glob(test_data_path  + "/*")

for path in imgs_paths:
    img = cv2.imread(path).astype(np.float32)
    img = cv2.resize(img, (227,227))
    img = img.transpose(2,1,0)
    imgs.append(img)
    #img = torch.from_numpy(img).expand(1,-1,-1,-1)
    #input(img.shape)
    #res = model(img)


imgs = torch.from_numpy(np.array(imgs))

res = model(imgs)
pred_label = torch.argmax(res, axis=1).tolist()

blind_res = []
labels = {0:'Healthy', 1: 'Unhealthy'}

for i in range(len(imgs_paths)):
    temp = [os.path.basename(imgs_paths[i]), labels[pred_label[i]]]
    blind_res.append(temp)

final_res = pd.DataFrame(blind_res, columns=["Test_Img_Name","Result"])

print(final_res.head())

final_res.to_csv("blind_set_results.csv",index=False)
    



# res = model(imgs)

# print(res)
