""""
Pattern Recongition Course
    Homework 5: Construct a CNN model

"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.metrics import accuracy_score
from tqdm import trange

num_classes = 10
class_index = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
               'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def horizonFlip(img):
    flip_img = np.zeros(img.shape, dtype=np.uint8)
    row_idx = 0
    for row in img:
        flip_img[row_idx] = row[::-1]
        row_idx += 1
    return flip_img


def colorJittering(img):
    color_img = np.zeros(img.shape, dtype=np.uint8)
    for r in range(len(img)):
        for c in range(len(img[r])):
            color_img[r][c][:2] = img[r][c][:2]  # Turn off 'B' channel
    return color_img


def noiseInjection(img):
    noise = np.random.normal(0, 8, img.shape)
    noise = np.array([int(n) for row in noise for col in row for n in col])
    noise = noise.reshape(img.shape)
    noise_img = img + noise
    noise_img[np.where(noise_img > 255)] = 255
    noise_img[np.where(noise_img < 0)] = 0
    return noise_img


def dataAugmentation(x, y):
    aug_number = 4
    x_new = np.zeros((len(x) * aug_number, 32, 32, 3), dtype=np.uint8)
    y_new = np.zeros((len(y) * aug_number, 1), dtype=np.int64)
    for i in trange(0, len(x) * aug_number, aug_number):
        index = i // aug_number
        y_new[i: i + aug_number] = y[index]
        x_new[i] = x[index]
        x_new[i + 1] = horizonFlip(x[index])
        x_new[i + 2] = colorJittering(x[index])
        x_new[i + 3] = noiseInjection(x[index])
    return x_new, y_new


class ImgDataset(Dataset):
    def __init__(self, x, y):
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
        # Input size = (3, 32, 32)
        self.conv1 = self.conv(3, 64, 3, 1, 1)  # (64,32,32)
        self.conv2 = self.conv(64, 64, 3, 1, 1)  # (64,32,32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # (64,16,16)
        self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv3 = self.conv(64, 128, 3, 1, 1)  # (128,16,16)
        self.conv4 = self.conv(128, 128, 3, 1, 1)  # (128,16,16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # (128,8,8)
        self.dropout2 = nn.Dropout2d(p=0.3)

        self.conv5 = self.conv(128, 128, 3, 1, 1)  # (128,8,8)
        self.conv6 = self.conv(128, 128, 3, 1, 1)  # (128,8,8)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # (128,4,4)
        self.dropout3 = nn.Dropout2d(p=0.4)

        # FC, input size = (16, 5, 5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu3 = nn.ReLU()
        self.dropout4 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.output = nn.Softmax(dim=1)

    def conv(self, in_c, out_c, kernel_size, stride=1, padding=0):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        return conv_layer

    def forward(self, x):
        out = self.dropout1(self.maxpool1(self.conv2(self.conv1(x))))
        out = self.dropout2(self.maxpool2(self.conv4(self.conv3(out))))
        out = self.dropout3(self.maxpool3(self.conv6(self.conv5(out))))
        out = torch.flatten(out, 1)  # from CNN to FCN
        out = self.output(self.fc2(self.dropout4(self.relu3(self.fc1(out)))))
        return out


if __name__ == "__main__":
    # Hyper-parameters
    batch_size = 50
    lr = 1e-4
    epochs = 100

    # Data
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    x_train_aug, y_train_aug = dataAugmentation(x_train, y_train)

    train_set = ImgDataset(x_train_aug, y_train_aug)
    test_set = ImgDataset(x_test, y_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Model
    torch.manual_seed(999)

    model = CNN().to(device)
    summary(model, (3, 32, 32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train model
    model.train()
    for epoch in trange(1, epochs + 1):
        avg_loss = 0
        for x, y in train_loader:
            xs = x.to(device)
            ys = y.to(device)

            optimizer.zero_grad()
            ps = model(xs)
            loss = criterion(ps, ys.squeeze())
            loss.backward()
            optimizer.step()

            avg_loss += loss

        print(f" Epoch {epoch}, {loss:.6f}")

    # Evaluate model
    model.eval()
    acc = 0
    for x, y in test_loader:
        xs = x.to(device)
        ys = y.to(device)

        ps = model(xs)

        pred_y = np.argmax(ps.to('cpu').detach().numpy(), 1)
        test_y = np.array(ys.to('cpu').flatten())
        acc += accuracy_score(pred_y, test_y, normalize=False)

    print("Accuracy of my model on test set: ", acc / len(test_set))

    torch.save(model.state_dict(), "./model/model_deeper.pt")
