import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt



def load_data(train_data_path):

    imgs = []

    labels = []

    classes = ["Healthy", "Unhealthy"]

    for label in classes:
        imgs_paths = glob.glob(train_data_path + label + "/*")
        for path in imgs_paths:
            img = cv2.imread(path).astype(np.float32)
            img = cv2.resize(img, (227,227))
            img = img.transpose(2,1,0)
            imgs.append(img)
            if label == 'Healthy':
                labels.append(0)
            else:
                labels.append(1)
    loader = DataLoader(list(zip(imgs,labels)), shuffle=True, batch_size=16)  ### need a better way to do this, len(imgs)
    return loader

def load_data_validation(valid_data_path):

    imgs = []

    labels = []

    classes = ["Healthy", "Unhealthy"]

    for label in classes:
        imgs_paths = glob.glob(valid_data_path + label + "/*")
        for path in imgs_paths:
            img = cv2.imread(path).astype(np.float32)
            img = cv2.resize(img, (227,227))
            img = img.transpose(2,1,0)
            imgs.append(img)
            if label == 'Healthy':
                labels.append(0)
            else:
                labels.append(1)
    loader = DataLoader(list(zip(imgs,labels)), shuffle=True, batch_size=len(imgs))  ### need a better way to do this, len(imgs)
    return loader


def get_accuracy(y_pred, y_gt):
    pred_label = torch.argmax(y_pred, axis=1)
    return (pred_label==y_gt).sum()/len(y_gt)

def create_plots(y1, y2, x, type):
    # Sample data
    x = list(range(x))
    
    # Plotting the data
    if type == "losses":
        plt.plot(x, y1, label='Training Loss')
        plt.plot(x, y2, label='Validation Loss')
    else:
        plt.plot(x, y1, label='Training accuracy')
        plt.plot(x, y2, label='Validation accuracy')

    # Setting labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Line Plot')

    # Setting legend
    plt.legend()

    # Displaying the plot
    plt.savefig(type + "2" + ".png")
    plt.clf()

