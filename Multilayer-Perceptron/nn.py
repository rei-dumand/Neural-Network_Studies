# Shadow-coding:
# https://github.com/Bot-Academy/NeuralNetworkFromScratch/blob/master/nn.py

from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

"""
w = weights
b = bias
i = input layer
h = hidden layer
o = output layer
l = label
for example: w_h_o = weights from hidden layer to output layer
"""


# Imports the images and labels
images, labels = get_mnist()
# Creates 15680 weight values, which correspond to the total
# number of connections between neurons of i & h.
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
# There are 200 connections between neurons of h & o
# Hence the need for only 200 weights.
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))

# a bias node is present between each layers,
# and provides the model more 'flexibility'.
# Think of how in f(x) = ax + y, y is what allows
# the linear function to be offset from the origin [0,0]
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

"""
This is just a mental note to see how zip works, how you make a list out of
    the resulting zip, and finally how you wrap this all in a numpy array.
The numpy array is a superior version of list, as it is contiguous in memory
    and therefore just runs faster than the regular list.
Additionally, numpy arrays ensure that the values in your lists are homogeneous
    while python lists allow you to store whatever you want regardless of the nature of its sibling values
--- --- ---
arr1 = [ "amit", "sumit", "vishal", "mita", "geets", "sujata", "kavita", "lalita", "somore" ]
arr2 = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = np.array(list(zip(arr1, arr2)))
print(result)
"""

learn_rate = 0.01
nr_correct = 0

# This is the sigmoid function that normalises our h activations.
def sig(val_pre):
    return 1 / (1 + np.exp(-val_pre))

epochs = range(3)
for epoch in epochs:
    # zip is a way to associate, at the nth position,
    # the nth item from the first to the nth item from the second list
    for img, l in zip(images, labels):
        # .shape returns the dimension of numpy arrays
        # So images is (60000, 784): 60,000 counts of 784 elements-long lists
        # We add (1,) to img.shape to convert it from a vector (one dimensional array)
        # to a matrix (two dimensional array)
        # This is required to avoid calculation errors
        # (you cannot do operations involving both 1-d & 2-d arrays)
        img.shape += (1,)
        l.shape += (1,)

        # to get the activation values on h, we do a matrix multiplication
        # of the i-h weights with the grayscale pixel values from our input data.
        h_pre = (w_i_h @ img) + b_i_h
        # We then normalise it using the sigmoid function
        h = sig(h_pre)

        # We then do the same to get the activations of our output layer.
        o_pre = (w_h_o @ h) + b_h_o
        o = sig(o_pre)

        # Cost calculation using mean-squared error
        # It computes the difference between the output activation
        # and the expected output for that output neuron
        # it squares it, sums the result of each neuron
        # and finally divides it by the number of neurons to obtain the average.
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)

        # This step is purely for monitoring.
        # .argmax returns the index of the highest value in a list
        # for l, it will always be the expected output since it is 1.
        # for o, if the index matches the index in l, then we can consider
        # that the model guessed correctly, and the conditional returns true
        # which in turns increment nr_correct by one for each correct guess.
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # backpropagation
        # This implementation doesn't look strictly correct.
        # I will have to investigate on a more rigorous implementation.
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # backpropagation from h to img
        # the derivative of a sigmoid function is S'(x) = S(x)(1 - S(x))
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28,28), cmap="Greys")
    img.shape += (1,)
    # Forward Propagation from i to h
    h_pre = w_i_h @ img.reshape(784, 1) + b_i_h
    h = sig(h_pre)
    # Then same from h to o
    o_pre = w_h_o @ h + b_h_o
    o = sig(o_pre)
    plt.title(f"I think this is a {o.argmax()}")
    plt.show()
