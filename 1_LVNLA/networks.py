import tensorflow as tf

class FFNet:

    # structure should be an array of sizes [in, w1, w2, ...., out]
    # activation_ids: 1 = identity, 2 = relu, 3 = sigmoid, 4 = tanh
    def __init__(self, structure, activation_id):
        self.structure = structure
        self.activation_id = activation_id
        self.session = None
        self.constructed = False

    def __del__(self):
        pass
    
    def construct(self):
        print("Constructing graph...")
        
        # init some vars
        self.weights = []
        self.biases = []
        self.layer_calcs = []
        self.layer_activations = []
        
        # set up inputs and outputs
        self.input = tf.placeholder(tf.float32, shape=[None, self.structure[1]])
        self.output = tf.placeholder(tf.float32, shape=[None, self.structure[-1]])

        prev_layer = self.input

        # set up hidden layers
        for i in range(1, len(structure)-1):
            self.weights.append(tf.Variable(tf.zeros([self.structure[i-1], self.structure[i]])))
            self.biases.append(tf.Variable(tf.zeros([self.structure[i]])))
            
            self.layer_calcs.append(tf.add(tf.matmul(prev_layer, self.weights[i]), self.biases[i]))

            if self.activation_id == 1:
                self.layer_activations.append(self.layer_calcs[i])
            elif self.activation_id == 2:
                self.layer_activations.append(tf.nn.relu(self.layer_calcs[i]))
            elif self.activation_id == 3:
                self.layer_activations.append(tf.nn.sigmoid(self.layer_calcs[i]))
            elif self.activation_id == 4:
                self.layer_activations.append(tf.nn.tanh(self.layer_calcs[i]))

            
        

    def train(self):
        pass

    def predict(self):
        pass



    
x_size = 2
y_size = 1

# network input and output placeholders
x = tf.placeholder("int", shape=[None, x_size])
y = tf.placeholder("int", shape=[None, x_size])

# weight initializations
