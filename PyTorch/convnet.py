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

REBUILD_DATA = True

class DogsVSCats ():
    IMG_SIZE = 50                   # This is the px dimension for both W & H. Aspect ratio is not preserved.
    CATS = "Data/PetImages/Cat"          # Filepaths.
    DOGS = "Data/PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    catCount = 0
    dogCount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            # tqdm is just a progress bar to give feedback at runtime
            for f in tqdm(os.listdir(label)):
                path = os.path.join(label, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.reize(img, (self.IMG_SIZE, self.IMG_SIZE))
                self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])