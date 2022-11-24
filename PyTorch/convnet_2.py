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
        x = x.flatten(start_dim=1)        # flattening happens here 
        # print(x.shape)                    # printing out to see what our shape is.
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

net = Net().to(device)
net.forward(torch.randn(1, 1, 50, 50).to(device))  # we run this once to pass dummy values to .forward
                                        # which allows us to see what the shape of our last conv is.


optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1   # We choose 10% of the training data to be used as validation data
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

# BATCH_SIZE = 100

# EPOCHS = 10

# for epoch in range(EPOCHS):
#     print(f"Starting Epoch {epoch}")
#     for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
#         # print(i, i + BATCH_SIZE)
#         batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
#         batch_y = train_y[i:i+BATCH_SIZE]

#         batch_X = batch_X.to(device)
#         batch_y = batch_y.to(device)

#         net.zero_grad()
#         outputs = net(batch_X)
#         loss = loss_function(outputs, batch_y)
#         loss.backward()
#         optimizer.step()

# correct = 0
# total = 0
# with torch.no_grad():
#     for i in tqdm(range(len(test_X))):
#         real_class = torch.argmax(test_y[i])
#         net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]
#         predicted_class = torch.argmax(net_out)
#         if predicted_class == real_class:
#             correct += 1
#         total += 1
#         pass
# print("Accuracy:", round(correct/total, 3))
# print("Loss:", loss)

def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True) / len(matches)
    loss = loss_function(outputs, y)
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    random_start = np.random.randint(len(test_X) - size)
    X, y = test_X[random_start: random_start + size], test_y[random_start:random_start + size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1, 1, 50, 50).to(device), y.to(device))
    return val_acc, val_loss

# val_acc, val_loss = test(size=32)
# print(val_acc, val_loss)
# val_acc1, val_loss1 = test(size=32)
# print(val_acc1, val_loss1)
# val_acc2, val_loss2 = test(size=32)
# print(val_acc2, val_loss2)

import time
MODEL_NAME = f"model-{int(time.time())}"

net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

print(MODEL_NAME)

def train():
    BATCH_SIZE = 50
    EPOCHS = 10
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i: i + BATCH_SIZE].view(-1, 1, 50, 50).to(device)
                batch_y = train_y[i: i + BATCH_SIZE].to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)
                if i % 50 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc), 2)},{round(float(loss), 4)},{round(float(val_acc), 2)},{round(float(val_loss), 4)}\n")

train()
                