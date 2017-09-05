import tensorflow as tf

class FFNN:

    def __init__(self, structure, activation=tf.identity, regression=False, learning_rate=.01):
        self.structure = structure
        self.activation = activation
        self.session = None
        self.constructed = False
        self.initialized = False
        self.regression = regression
        self.learning_rate = learning_rate
