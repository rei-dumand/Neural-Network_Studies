"""
A convolution is also called a kernel.
neural networks work on numbers.

Pooling is just taking the max value of a sample window
make sure your dataset is balanced:
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = False

class DogsVSCats ():
    IMG_SIZE = 50                   # This is the px dimension for both W & H. Aspect ratio is not preserved.
    CATS = "data/PetImages/Cat"          # Filepaths.
    DOGS = "data/PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    catCount = 0
    dogCount = 0

    def make_training_data(self):
        for label in self.LABELS:
            # print(label)
            # tqdm is just a progress bar to give feedback at runtime
            for f in tqdm(os.listdir(label)):
                try: 
                    path = os.path.join(label, f)
                    # print(path)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    if label == self.CATS:
                        self.catCount += 1
                    elif label == self.DOGS:
                        self.dogCount += 1
                except Exception as e:
                    pass
    
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: ", self.catCount)
        print("Dogs: ", self.dogCount)

if REBUILD_DATA == True:
    dogsvscats = DogsVSCats()
    dogsvscats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)

import matplotlib.pyplot as plt

# # To check that the data has been correctly loaded & processed:
# print(training_data[30])
# plt.imshow(training_data[30][0])
# plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool1 = nn.MaxPool2d((2,2))
        self.pool2 = nn.MaxPool2d((2,2))
        self.pool3 = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(512, 512)  # notice how conv3 has 128 out channels, yet fc1 takes in 512 inputs
                                        # to find that value without doing manual calcs, you can flatten x
                                        # after the convolutions, before passing it to the dense layers.
                                        # then print out the shape of x, and it will give you smth like this:
                                        # "torch.Size([1, 512])". Here, meaning that there are 512 inputs/neurons.
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        # x = x.flatten(start_dim=1)        # flattening happens here 
        # print(x.shape)                    # printing out to see what our shape is.
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = Net()
# net.forward(torch.randn(1, 1, 50, 50))    # we run this once to pass dummy values to .forward
#                                           # which allows us to see what the shape of our last conv is.


optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1   # We choose 10% of the training data to be used as validation data
val_size = int(len(X)*VAL_PCT)
# print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

# print(len(train_X))
# print(len(test_X))

BATCH_SIZE = 100

EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        print(i, i+BATCH_SIZE)
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss)