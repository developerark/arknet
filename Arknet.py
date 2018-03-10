import numpy as np 
from Layer import Layer

class Arknet:
    # Initialize the network
    def __init__(self):
        self.__layers = []
        self.__learningRate = 0.3
        self.__weights = None
        pass
    
    # Set learning rate
    def setLearningRate(self, learningRate):
        self.__learningRate = learningRate

    # Append layer to the network
    def appendLayer(self, layer):
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
        inputs = np.array(inputs, ndim=2).T
        targets = np.array(targets, ndim=2).T 
        
        self.__layers[0].outputs = inputs
        for i in range(1, len(self.__layers)):
            X_i = np.dot(self.__weights[i - 1], self.__layers[i-1].outputs)
            self.__layers[i].outputs = self.__layers[i].activationFunction(X_i)
        return self.__layers[-1].outputs

        # Calculating the error = target - output
        self.__layers[-1].errors = targets - finalOutputs
        for i in range(len(self.__layers)-2, -1, -1):
            self.__weights[i] += self.__learningRate * np.dot((self.__layers[i+1].errors * self.__layers[i+1].outputs * (1.0 - self.__layers[i+1].outputs)), np.transpose(self.__layers[i].outputs))
        

    # Query the neural network
    def query(self, inputs):
        self.__layers[0].outputs = inputs
        for i in range(1, len(self.__layers)):
            X_i = np.dot(self.__weights[i - 1], self.__layers[i-1].outputs)
            self.__layers[i].outputs = self.__layers[i].activationFunction(X_i)
        return self.__layers[-1].outputs


if __name__ == "__main__":

    layer1 = Layer(3)
    layer2 = Layer(3)
    layer3 = Layer(3)

    network = Arknet()
    network.appendLayer(layer1)
    network.appendLayer(layer2)
    network.appendLayer(layer3)

    print(network.query([1.0, 0.5, -1.5]))

