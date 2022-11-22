
import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights1 = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_output = np.dot(inputs, np.array(weights1).T) + biases1
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
print(layer1_output)
print(layer2_output)

# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0



# oneD = np.array([1, 2, 3, 4])
# print(oneD.shape)

# twoD = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# print(twoD.shape)

# threeD = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print(threeD.shape)
