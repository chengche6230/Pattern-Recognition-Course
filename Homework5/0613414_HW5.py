""""
Pattern Recongition Course
    Homework 5: Construct a CNN model

"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import trange

num_classes = 10
class_index = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
               'dog': 5, 'frog': 6,'horse': 7,'ship': 8, 'truck': 9}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImgDataset(Dataset):
    def __init__(self, x_file, y_file):
        x = np.load(x_file)
        y = np.load(y_file)
        
        x = x.astype('float32')
        x /= 255
        
        self.x = torch.from_numpy(x).permute(0, 3, 1, 2)
        self.y = torch.from_numpy(y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # CNN 1, input size = (3, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)  # (3,32,32)->(32,28,28)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # (32,28,28)->(32,14,14)
        
        # CNN 2, input size = (32, 14, 14)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5)  # (32,14,14)->(16,10,10)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # (16,10,10)->(16,5,5)
        
        # FCN, input size = (16, 5, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.output = nn.Softmax(dim=1)
        
    def cnn(self, in_c, out_c, kernel_size):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
        )
        return conv_layer
    
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = torch.flatten(out, 1)  # from CNN to FCN
        out = self.output(self.fc2(self.relu3(self.fc1(out))))
        return out

if __name__ == "__main__":
    # Hyper-parameters
    batch_size = 25
    lr = 1e-4
    epochs = 100
    momentum = 0.9
    
    # Data
    train_dataset = ImgDataset("x_train.npy", "y_train.npy")
    test_dataset = ImgDataset("x_test.npy", "y_test.npy")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Model related 
    torch.manual_seed(999)
    
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    model.train()
    for epoch in trange(1, epochs + 1):
        print(f"Epoch:{epoch}")
        for x, y in train_loader:
            xs = x.to(device)
            ys = y.to(device)
            
            optimizer.zero_grad()
            ps = model(xs)
            loss = criterion(ps, ys.squeeze())
            loss.backward()
            optimizer.step()
    
    # Predict
    model.eval()
    acc = 0
    for x, y in test_loader:
        xs = x.to(device)
        ys = y.to(device)
        
        ps = model(xs)
        pred_y = np.argmax(ps.to('cpu').detach().numpy(), 1)
        test_y = np.array(ys.to('cpu').flatten())
        acc += accuracy_score(pred_y,test_y, normalize=False)
        #print(ps.to('cpu').detach().numpy())
        #print(test_y)
        #print(acc)
    print("Accuracy of my model on test set: ", acc / len(test_dataset))