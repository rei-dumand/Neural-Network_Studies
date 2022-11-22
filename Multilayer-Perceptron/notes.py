"""
A multilayer perceptron is the most basic form of machine learning using a neural network.

neural ==> neuron ==> a thing that holds a number in our example

Here, each pixel is a neuron, so 28px x 28 px = 784px. So 784 neurons
    Each neuron holds a value that corresponds to the grayscale value of the pixel in question.
    That value is called the neuron's 'activation', and is normalised between 0 and 1.

The input layer will have as many neurons as there are pixels in the image.
The output layer will only have as many neurons as there are possible outcomes. 
    In our case because we are examining which digit the image represents, we have 10 possible options:
    0 - 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9
    Which means that out output layer will only have 10 neurons.

Hidden layers are all the layers in between.

Activations in one layer inform the activations of the next layer, and so on.

When we say that the model learns, we are essentially trying to get it to find 
    the right weights and biases to solve the problem at hand.

Sigmoid function is used to normalise the summed products of weight 

You assign weights between each of the connections between the input layer and first hidden layer.
You then compute the summed products of each activation with its respective assigned weight.
Because that weighted sum can be a very large value, 
you use a function to normalise the result between 0 and 1.
This value is simply a way to describe how positive, or activated, that neuron is.

In a neural network, you also apply a bias before normalising your resulting activation.
Why the bias? Currently I'm not sure, so I'll need to find a separate resource for that.
All I know that the bias is seemingly random, and controls the ability for the neuron to be activated. 

Neurons are better thought as functions, as they take in the outputs of all the neurons in the previous layer.

Nowadays, sigmoid functions are deprecated in neural networks.
    Rectified linear unit (or ReLU) is preferred.
    In simple terms, in ReLU, below a certain threshold, the neuron is not activated at all.
    And above that threshold: f(a) = a

Initially, you assign random weights and biases. This is just to get you started and get an output.
The result will be as expected: random.
To tell the algorithm how it performed, you define a cost function, that tells it what the goal is.
Formula is the SUM OF (result activation of output neuron - expected activation of output neuron)²
That sum will be a large value at first, and should become smaller as the network outputs estimates
that more closely align with the expected output.

Hence, a supervised neural network is one where we know what the expected output is, and where we can
define a cost function to guide the network.

Once you average the cost of all the training data, that value gives an overview of how good or bad the
network is. The aim will be to get that value as close to zero as possible.

To recap, first a neural network has a function:
    input: each individual pixel of the training image.
    output: 10 numbers representing the 10 possible digits that image represents.
    parameters: 13,002 weights/biases.

The cost function is a layer on top of that function:
    input: 13,002 weights/biases
    output: 1 number (called the cost)
    parameters: all of the training data.

When we say that a network is learning, it's just minimising a cost function.

The gradient vector of our cost function is a way to define the relative importance of each weight/bias.
    Should this weight/bias change a little, or a lot? and in which direction?

Couple of new words I have to make sense of:
Backpropagation is the algorithm that determines how a single training example would like to nudge the
    weight/bias of the input neurons.
    This nudge is quantified both in terms of direction, but also in terms of proportion, 
    so as to decrease the cost as quickly as possible.

A gradient descent step is the value given by the series of calculations made during the backpropagation.
But because this calculation is expensive, you don't want to obtain a gradient step with every single
    training data every time.
    Instead, you divide your training data randomly into batches, and each iteration uses one of these batches
    to compute the gradient descent step.
    This is called stochastic gradient step.

 quantifying the importance of each weight/bias within input neurons
    on the output neuron, so that we can determine the direction and proportion of change needed on each
    weight/bias in order to get a series of output that is closer to our expected output.

"""


"""
Linear Algebra and matrices
x = 1 & y = 2

[0 -1][x]  ==> x*0 + y*(-1)   ==> -2
[1  0][y]  ==> x*1 + y*0      ==> 1

for x = 1 & y = 1

[1  3][x] ==> x*1 + y*3       ==> 4
[2  1][y] ==> x*2 + y*1       ==> 3

"""
