import tensorflow as tf

class FFNet:

    # structure should be an array of sizes [in, w1, w2, ...., out]
    # activation_ids: 1 = identity, 2 = relu, 3 = sigmoid, 4 = tanh
    def __init__(self, structure, activation_id, regression=True, learning_rate=.01):
        self.structure = structure
        self.activation_id = activation_id
        self.session = None
        self.constructed = False
        self.initialized = False
        self.regression = regression
        self.learning_rate = learning_rate

    def __del__(self):
        if self.initialized:
            self.session.close()
    
    def construct(self):
        print("Constructing graph...")
        
        # init some vars
        self.weights = []
        self.biases = []
        self.layer_calcs = []
        self.layer_activations = []
        
        # set up inputs and outputs
        self.input = tf.placeholder(tf.float32, shape=[None, self.structure[0]])
        self.output_ = tf.placeholder(tf.float32, shape=[None, self.structure[-1]])

        prev_layer = self.input

        # set up hidden layers
        for i in range(1, len(self.structure)):
            self.weights.append(tf.Variable(tf.zeros([self.structure[i-1], self.structure[i]])))
            self.biases.append(tf.Variable(tf.zeros([self.structure[i]])))

            layer_i = i - 1
            
            self.layer_calcs.append(tf.add(tf.matmul(prev_layer, self.weights[layer_i]), self.biases[layer_i]))

            if self.activation_id == 1 or i == len(self.structure) - 1: # leave final output alone (no activation)
                self.layer_activations.append(self.layer_calcs[layer_i])
            elif self.activation_id == 2:
                self.layer_activations.append(tf.nn.relu(self.layer_calcs[layer_i]))
            elif self.activation_id == 3:
                self.layer_activations.append(tf.nn.sigmoid(self.layer_calcs[layer_i]))
            elif self.activation_id == 4:
                self.layer_activations.append(tf.nn.tanh(self.layer_calcs[layer_i]))
            else: print("ERROR - invalid activation id '" + str(self.activation_id) + "'")
            
            prev_layer = self.layer_activations[layer_i]

        self.output = self.layer_activations[-1]

        # regression cost
        if self.regression: self.cost = tf.reduce_mean(tf.squared_difference(self.output, self.output_))
        else: self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.output_))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.tester = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.output), self.output_), tf.float32))
        #self.tester = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.output), self.output_), tf.int32))
        #self.tester = self.cost

                
        self.constructed = True
        print("Graph constructed!")
    
    def initialize_session(self):
        if not self.initialized:
            if not self.constructed: self.construct()
            
            print("Initializing...")
            self.session = tf.InteractiveSession()
            self.session.run(tf.global_variables_initializer())
            self.initialized = True
            print("Initialized!")
            tf.summary.FileWriter("summary", self.session.graph)
            
    # thanks to https://stackoverflow.com/questions/42613747/tensorflow-splitting-training-data-to-batches
    def train(self, graph_input, graph_target, epochs, batchsize):
        if not self.initialized: self.initialize_session()
        
        # divide into batches
        print("Batchifying...")
        batches = []
        for i in range(int(len(graph_input)/batchsize)):
            inputs = graph_input[i*batchsize:(i+1)*batchsize]
            targets = graph_target[i*batchsize:(i+1)*batchsize]
            batches.append([inputs, targets])
        print("Batchified!")
    
        print("Training...")
        for i in range(epochs):
            print("Epoch " + str(i))
            for j in range(len(batches)):
                self.optimizer.run(feed_dict={self.input: batches[j][0], self.output_: batches[j][1]})
                if j == (len(batches) - 1): self.test(graph_input, graph_target)

        print("Training complete")
        self.test(graph_input, graph_target)

    def test(self, graph_input, graph_target):
        accuracy = self.tester.eval(feed_dict={self.input: graph_input, self.output_: graph_target})
        print("Accuracy: " + str(accuracy))

    def predict(self, graph_input):
        return self.output.eval(feed_dict={self.input: graph_input})
