import numpy as np 
from Layer import Layer
import sys

class Arknet:
    # Initialize the network
    def __init__(self):
        self.__layers = []
        self.__learningRate = 0.05
        self.__weights = None
        pass
    
    # Set learning rate
    def setLearningRate(self, learningRate):
        self.__learningRate = learningRate

    # Append layer to the network
    def appendLayers(self, *layers):
        for layer in layers:
            self.__layers.append(layer)
        self.__weights = [None] * (len(self.__layers) - 1)
        self.__initializeWeights()

    # Remove a certail layer
    def removeLayer(self, layer):
        try:
            self.__layers.remove(layer)
        except Exception as error:
            print(error)
        self.__weights = [None] * (len(self.__layers) - 1)
        self.__initializeWeights()

    # Initialize weights
    def __initializeWeights(self):
        if len(self.__layers) < 1:
            print("[-] Not enough layers to initialize weights")
            exit(0)
        
        # Start initializing the weights
        for i in range(0, len(self.__weights)):
            left, right = i, i + 1
            self.__weights[i] = np.random.normal(0.0, pow(self.__layers[left].numberOfNodes, -0.5), (self.__layers[right].numberOfNodes, self.__layers[left].numberOfNodes))

    # Train the network
    def train(self, inputs, targets):
        # Convert inputs to 2d array
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T 
        
        self.__layers[0].outputs = inputs
        for i in range(1, len(self.__layers)):
            X_i = np.dot(self.__weights[i - 1], self.__layers[i-1].outputs)
            self.__layers[i].outputs = self.__layers[i].activationFunction(X_i)

        # Calculating the error = target - output
        self.__layers[-1].errors = targets - self.__layers[-1].outputs
        for i in range(len(self.__layers)-2, -1, -1):
            self.__layers[i].errors = np.dot(self.__weights[i].T, self.__layers[i+1].errors)
            self.__weights[i] += self.__learningRate * np.dot((self.__layers[i+1].errors * self.__layers[i+1].outputs * (1.0 - self.__layers[i+1].outputs)), np.transpose(self.__layers[i].outputs))
        

    # Query the neural network
    def query(self, inputs):
        self.__layers[0].outputs = inputs
        for i in range(1, len(self.__layers)):
            X_i = np.dot(self.__weights[i - 1], self.__layers[i-1].outputs)
            self.__layers[i].outputs = self.__layers[i].activationFunction(X_i)
        return self.__layers[-1].outputs

    # Destructor
    def __del__(self):
        pass

def trainMNIST(network, outputNodes):
    fob = open("datasets/mnist_train.csv", 'r')
    lines = fob.readlines()
    fob.close()
    totalLines = len(lines)
    count = 0
    for line in lines:
        values = line.rstrip().split(',')
        inputs = ((np.asfarray(values[1:])/255.0) * 0.99) + 0.01
        targets = np.zeros(outputNodes) + 0.01
        targets[int(values[0])] = 0.99
        network.train(inputs, targets)
        count += 1
        sys.stdout.write("\r\tProgress: %6.2f" % ((count * 100.0) / totalLines))
        sys.stdout.flush()
    print

def testMNIST(network):
    fob = open("datasets/mnist_test.csv", 'r')
    lines = fob.readlines()
    fob.close()
    totalCorrect = 0
    totalFalse = 0
    total = len(lines)
    for line in lines:
        values = line.rstrip().split(',')
        inputs = ((np.asfarray(values[1:])/255.0) * 0.99) + 0.01
        groundTruth = int(values[0])
        output = network.query(inputs)
        prediction = output.argmax()
        #print("Ground Truth: %i, Prediction: %i" % (groundTruth, prediction))
        if groundTruth == prediction:
            totalCorrect += 1
        else:
            totalFalse += 1
    print("\tAccuracy: %6.2f" % ((totalCorrect * 100.0)/total))
    #print("Incorrect: %6.2f" % ((totalFalse * 100.0)/total))
    

if __name__ == "__main__":
    network = Arknet()
    network.appendLayers(Layer(784), Layer(20), Layer(40), Layer(80),  Layer(10))

    for i in range(30):
        print("Epoch: %i" % (i + 1))
        trainMNIST(network, 10)
        testMNIST(network)
