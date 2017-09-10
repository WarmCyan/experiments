import tensorflow as tf

def var_summary(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        #with tf.name_scope('stddev'):
        #    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        #tf.summary.scalar('stddev', stddev)
        #tf.summary.scalar('max', tf.reduce_max(var))
        #tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

class FNN:

    def __init__(self, structure, logdir, name, activation=tf.identity, regression=False, learning_rate=.01):
        self.structure = structure
        self.logdir = logdir
        self.name = name
        self.activation = activation
        self.session = None
        self.constructed = False
        self.regression = regression
        self.learning_rate = learning_rate

    def __del__(self):
        self.stopSession()

    def stopSession(self):
        self.session.close()

    def construct(self):
        print("Constructing graph...")

        self.graph = tf.Graph()

        self.weights = []
        self.biases = []
        self.layer_calcs = []
        self.layer_activations = []

        with self.graph.as_default():

            # set up inputs and outputs
            self.input = tf.placeholder(tf.float32, shape=[None, self.structure[0]], name='input')
            self.output_ = tf.placeholder(tf.float32, shape=[None, self.structure[-1]], name='output_')

            # set up hidden layers
            prev_layer = self.input
            for i in range(1, len(self.structure)-1):

                current_layer = i-1
                next_layer = i
                
                with tf.name_scope('hiddenlayer' + str(i)):
                    with tf.name_scope('weights'):
                        print("Connecting " + str(self.structure[current_layer]) + " to " + str(self.structure[next_layer]))
                        #w = tf.Variable(tf.random_normal([self.structure[current_layer], self.structure[next_layer]]))
                        #self.weights.append(tf.Variable(tf.random_normal([self.structure[current_layer], self.structure[next_layer]])))
                        self.weights.append(tf.Variable(tf.random_normal([self.structure[current_layer], self.structure[next_layer]])))
                        var_summary(self.weights[-1])
                    with tf.name_scope('bias'):
                        b = tf.Variable(tf.zeros([self.structure[next_layer]]))
                        var_summary(b)
                        self.biases.append(b)
                    with tf.name_scope('preact'):
                        preact = tf.add(tf.matmul(prev_layer, self.weights[current_layer]), self.biases[current_layer])
                        var_summary(preact)
                        self.layer_calcs.append(preact)
                    with tf.name_scope('layer_out'):
                        act = self.activation(self.layer_calcs[current_layer])
                        var_summary(act)
                        self.layer_activations.append(act)

                prev_layer = self.layer_activations[current_layer]

            # set up output layer
            with tf.name_scope('outlayer'):
                with tf.name_scope('weights'):
                    w = tf.Variable(tf.random_normal([self.structure[-2], self.structure[-1]]))
                    var_summary(w)
                    self.weights.append(w)
                with tf.name_scope('bias'):
                    b = tf.Variable(tf.zeros([self.structure[-1]]))
                    var_summary(b)
                    self.biases.append(b)
                with tf.name_scope('layer_out'):
                    out = tf.add(tf.matmul(prev_layer, self.weights[-1]), self.biases[-1])
                    if self.regression:
                        self.output = out
                    else:
                        self.output = tf.nn.softmax(out)
                    #out = tf.add(tf.matmul(prev_layer, w), b)
                    var_summary(self.output)

            with tf.name_scope('loss'):
                if self.regression:
                    self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.output_, predictions=self.output))
                    tf.summary.scalar('loss', self.loss)
                else:
                    self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.output_, logits=self.output))
                    tf.summary.scalar('cross_entropy', self.cross_entropy)

            with tf.name_scope('training'):
                if self.regression:
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                else:
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

            with tf.name_scope('testing'):
                with tf.name_scope('correct'):
                    self.correct = tf.equal(tf.round(self.output), tf.round(self.output_))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

            self.session = tf.Session(graph=self.graph)
            
            self.merged_summaries = tf.summary.merge_all()
            self.session.run(tf.global_variables_initializer())
            self.train_writer = tf.summary.FileWriter(self.logdir + self.name, self.session.graph)
            self.test_writer = tf.summary.FileWriter(self.logdir + self.name)
            self.train_writer.add_graph(self.session.graph)
            self.test_writer.add_graph(self.session.graph)

        self.constructed = True
        print("Constructed!")
                
    def train(self, graph_inputs, graph_targets, epochs, batchsize):
        if not self.constructed: self.construct()
        
        # divide into batches
        print("Batchifying...")
        batches = []
        for i in range(int(len(graph_inputs)/batchsize)):
            inputs = graph_inputs[i*batchsize:(i+1)*batchsize]
            targets = graph_targets[i*batchsize:(i+1)*batchsize]
            batches.append([inputs, targets])
        print("Batchified!")
        
        # train
        for i in range(epochs):
            for j in range(len(batches)):
                summary, _ = self.session.run([self.merged_summaries, self.train_op], feed_dict={self.input:batches[j][0], self.output_:batches[j][1]})
                self.train_writer.add_summary(summary, i*1000 + j)
            summary, acc = self.session.run([self.merged_summaries, self.accuracy], feed_dict={self.input:inputs, self.output_:targets})
            self.test_writer.add_summary(summary, i*10000)
            print(acc)

    def predict(self, graph_inputs):
        result = self.session.run([self.output], feed_dict={self.input: graph_inputs})
        return result
