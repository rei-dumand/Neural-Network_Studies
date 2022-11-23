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
