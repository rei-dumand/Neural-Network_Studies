import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

train = datasets.MNIST("", train=True, download=True,
                        transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=False,
                        transform= transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

"""
If you are confused by how PyTorch delivers data, just print one batch to understand the data's shape:
"""
# for data in trainset:
    # print(data)           # we have [tensor(*list of* list of images), tensor(list of labels)]
    # print(len(data[0]))   # 10 items because batch_size = 10
    # print(data[0].shape)  # note that the images' tensor's shape is:
    #                           # 10 (items or images) of 
    #                           #  1 (wraps the list... why...) of
    #                           # 28 (px/row) by
    #                           # 28 (px/column)
    # print(len(data[1]))   # also 10 items because for 10 images you should have 10 labels
    # print(data[1])        # the labels' tensor is just a list of 10 items

    ## Note that every batch is different each new run because the data is shuffled on-load.
    # break

"""
What if I want to make sure my data matches my label? Matplotlib 😏
"""
# for data in trainset:
#     image, label = data[0][0], data[1][0] # we find the very first image/label pair provided
#     print(label)
#     print(image.shape)
#     image = image.view(28, 28)      # imshow takes in an image, which is a matrix of pixels (2D array)
#                                     # hence the shape conversion from [1, 28, 28] to [28, 28] using torch view()
#     plt.imshow(image)       # next step is loading the image into matplotlib
#     plt.show()              # and finally display it
#     break

"""
Finally if I want to see if my dataset is balanced:
"""
# # first we count all instances of each possible label
# total = 0   # keeps a count of the total number of data points (images) in our dataset
# counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
# for data in trainset:
#     labels = data[1]                    # we only need our labels
#     for label in labels:
#         counter_dict[int(label)] += 1   # lookup in our dictionary for the label value, and increment it.
#         total += 1
# # and then I just need to print out the occurence rate of each label value 
# for i in counter_dict:
#     print(f"{i}: {counter_dict[i]/total*100}")

class Net(nn.module):
    def __init__(self):
        super().__init__()
