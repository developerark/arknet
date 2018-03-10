import scipy.special

class Layer:
    # Initialize the layer
    def __init__(self, numberOfNodes):
        self.numberOfNodes = numberOfNodes
        self.activationFunction = lambda x: scipy.special.expit(x)
        self.outputs = None
        self.errors = None
