import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import os
from natsort import natsorted
import os
from torch.utils.data import random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix


### Data for Auto-Encoder
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = listdir_nohidden(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image
    
### Model for Auto-Encoder
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=0), ##124 output
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0), ##61 output
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=0), ## 30 output
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=0), ## 14 output
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(14 * 14 * 32, 256),
            nn.ReLU(True),
            nn.Linear(256, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 14 * 14 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 14, 14))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding = 0, output_padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=0, output_padding = 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=0, output_padding = 1),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        # x = torch.sigmoid(x)
        return x

### Evaluation function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        val_loss = 0
        for i, image_batch in enumerate(dataloader):
            image_batch = image_batch.to(device)
            encoded_data = encoder(image_batch)
            decoded_data = decoder(encoded_data)
            loss = loss_fn(decoded_data, image_batch)
            val_loss += loss.data
    return val_loss
### Training helper function for autoencoder
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer, val_loader):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_losses = []
    validation_losses = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for i, image_batch in enumerate(dataloader):
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_losses.append(loss.detach().cpu().numpy())
        if i%5 == 0:
            val_loss = test_epoch(encoder, decoder, device, val_loader, loss_fn)
            print("Validation loss ", val_loss.data)
            validation_losses.append(val_loss.data)

    return train_losses, validation_losses

def train_auto_encoder(lr = 0.001, num_epochs = 7, hidden_dimension_size = 128, batch_size = 4, data_folder  = './healthy_leafs_data/'
):
    torch.manual_seed(2)
    loss_fn = torch.nn.MSELoss()
    encoder = Encoder(encoded_space_dim=hidden_dimension_size)
    decoder = Decoder(encoded_space_dim=hidden_dimension_size)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    encoder.to(device)
    decoder.to(device)
    print("Encoder ", encoder)
    print("Decoder ", decoder)
    transform = T.Compose([T.Resize(size = (250,250)), T.ToTensor()])
    ImageDataSet = CustomDataSet(data_folder, transform)
    train,val,test = random_split(ImageDataSet, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train , batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test,batch_size = batch_size)
    allepoch_train_losses = []
    allepoch_valid_losses = []
    for epoch in range(num_epochs):
        print("Epoch ", epoch+1)
        train_losses, valid_losses = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim, val_loader)
        print("Mean training loss per batch ", sum(train_losses)/len(train_losses))
        print("Mean validation loss ", sum(valid_losses)/len(valid_losses))
        allepoch_train_losses.append(train_losses)
        allepoch_valid_losses.append(valid_losses)
    torch.save(encoder.state_dict(), './testsave_encoder.pt')

    return allepoch_train_losses, allepoch_valid_losses


### For the classifier
class Classifier(nn.Module):
    def __init__(self, encoded_space_dim, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(encoded_space_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_classifier(classifier, lr = 0.001, num_epochs = 40, data_folder = './beans/', batch_size = 16, momentum = 0.9):
    train_folder = data_folder+'train/'
    valid_folder = data_folder+'valid/'
    transform = T.Compose([T.Resize(size = (250,250)), T.ToTensor()])
    train_dataset = datasets.ImageFolder(train_folder, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_folder, transform=transform)
    batch_size = 16
    train_loader = DataLoader(train_dataset , batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size)
    train_losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=momentum)
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_losses.append(loss.item())
            if i % 20 == 19:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return train_losses

def flatten(l):
    return [item for sublist in l for item in sublist]
def evaluate(classifier, criterion, data_folder ='./beans/valid/'):
    batch_size = 16
    transform = T.Compose([T.Resize(size = (250,250)), T.ToTensor()])
    test_dataset = datasets.ImageFolder(data_folder, transform=transform)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    test_losses = []
    preds = []
    true_labels = []
    with torch.no_grad():
      for i, data in enumerate(testloader):
        images, labels = data
        outputs = classifier(images)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs,labels)
        test_losses.append(loss.item())
        preds.append(predictions.tolist())
        true_labels.append(labels.tolist())

    preds = flatten(preds)
    true_labels = flatten(true_labels)
    accuracy = accuracy_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds)

    return test_losses, accuracy, cm

## get blind results
def make_prediction(classifier, data_folder = './beans/test/'):
    transform = T.Compose([T.Resize(size = (250,250)), T.ToTensor()])
    ImageDataSet = CustomDataSet(data_folder, transform)
    batch_size = 16
    data_loader = DataLoader(ImageDataSet , batch_size=batch_size)
    preds = []
    with torch.no_grad():
      for i, data in enumerate(data_loader):
        images = data
        outputs = classifier(images)
        _, predictions = torch.max(outputs, 1)
        preds.append(predictions.tolist())
    preds = flatten(preds)
    return preds

def save_blind(preds, data_folder = './beans/test/'):
    all_imgs = listdir_nohidden(data_folder)
    image_names = natsorted(all_imgs)
    df = pd.DataFrame(list(zip(image_names, preds)),columns = ['image names', 'predictions'])
    df.to_csv('./blind_results_sample.csv')
    return df

if __name__ == '__main__':
    # train_auto_encoder()

    ## training classifier
    # encoder = Encoder(128)
    # encoder.load_state_dict(torch.load('./encoder.pt'))
    # classifier = Classifier(128, encoder)
    # train_classifier(classifier)
    # ## test classifier
    # criterion = nn.CrossEntropyLoss()
    # test_losses, accuracy, cm = evaluate(classifier, criterion)
    # print("accuracy ", accuracy)
    # print("Confusion Matrix ", cm)
    # torch.save(classifier.state_dict(), './classifier.pt')

    ## get results on blind dataset
    # encoder = Encoder(128)
    # classifier = Classifier(128, encoder)
    # classifier.load_state_dict(torch.load('./classifier.pt'))
    # preds = make_prediction(classifier)
    # save_blind()

    
