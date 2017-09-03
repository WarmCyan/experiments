import tensorflow as tf

def var_summary(var):
    with tf.name_scope('summaries'):
        tf.summary.histogram('hist', var)

def run(inputs, targets, in_size=2, out_size=2, h_size=5, h_lcount=1, activation=tf.identity, learning_rate=.05, regression=True, epochs=100, batchsize=10, name=''):

    # input placeholders
    x = tf.placeholder(tf.float32, [None, in_size], name='x')
    y_ = tf.placeholder(tf.float32, [None, out_size], name='y_')

    # hidden layers
    hin = x
    hin_size = in_size
    maxi = 0
    for i in range(h_lcount-1):
        with tf.name_scope('hiddenlayer' + str(i)):
            with tf.name_scope('weights'):
                w = tf.Variable(tf.random_normal[hin_size, h_size])
                var_summary(w)
            with tf.name_scope('bias'):
                b = tf.Variable(tf.zeros([h_size]))
                var_summary(b)
            with tf.name_scope('preact'):
                preact = tf.add(tf.matmul(hin, w), b)
                var_summary(preact)
            with tf.name_scope('layer_out'):
                act = activation(preact)
                var_summary(act)
        hin = act
        hin_size = h_size
        maxi = i
    
    # final hidden layer
    maxi += 1       
    with tf.name_scope('hiddenlayer' + str(maxi)):
        with tf.name_scope('weights'):
            w = tf.Variable(tf.random_normal[hin_size, out_size])
            var_summary(w)
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([out_size]))
            var_summary(b)
        with tf.name_scope('preact'):
            preact = tf.add(tf.matmul(hin, w), b)
            var_summary(preact)
        with tf.name_scope('layer_out'):
            act = activation(preact)
            var_summary(act)

    if regression:
        loss = tf.losses
        
