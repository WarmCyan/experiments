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
    for i in range(h_lcount):
        with tf.name_scope('hiddenlayer' + str(i)):
            with tf.name_scope('weights'):
                w = tf.Variable(tf.random_normal([hin_size, h_size]))
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
    
    # final hidden layer
    with tf.name_scope('outlayer'):
        with tf.name_scope('weights'):
            w = tf.Variable(tf.random_normal([hin_size, out_size]))
            var_summary(w)
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([out_size]))
            var_summary(b)
        with tf.name_scope('layer_out'):
            if not regression:
                out = tf.nn.softmax(tf.add(tf.matmul(hin, w), b))
            else:
                out = tf.add(tf.matmul(hin, w), b)
            var_summary(out)

    # loss function
    with tf.name_scope('loss'):
        if regression:
            loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, logits=out))
            tf.summary.scalar('loss', loss)
        else:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))
            tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('training'):
        if regression:
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else:
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    with tf.name_scope('testing'):
        with tf.name_scope('correct'):
            correct = tf.equal(tf.round(out), tf.round(y_))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    sess = tf.InteractiveSession()

    merged = tf.summary.merge_all()
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter("summary/" + name, sess.graph)
    test_writer = tf.summary.FileWriter("summary/" + name)
    train_writer.add_graph(sess.graph)
    test_writer.add_graph(sess.graph)
    
    batches = []
    for i in range(int(len(inputs)/batchsize)):
        inp = inputs[i*batchsize:(i+1)*batchsize]
        tar = targets[i*batchsize:(i+1)*batchsize]
        batches.append([inp, tar])
        
    for i in range(epochs):
        for j in range(len(batches)):
            summary, _ = sess.run([merged, train_op], feed_dict={x:batches[j][0], y_:batches[j][1]})
            train_writer.add_summary(summary, i*1000 + j)
        summary, acc = sess.run([merged, accuracy], feed_dict={x:inputs, y_:targets})
        test_writer.add_summary(summary, i*10000)
        print(acc)
        
        result = sess.run([out], feed_dict={x:[[0,0]], y_:[[0,1]]})
        print(result)
        result = sess.run([out], feed_dict={x:[[0,1]], y_:[[1,0]]})
        print(result)
