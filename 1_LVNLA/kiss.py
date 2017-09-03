import tensorflow as tf


def var_summary(var):
    with tf.name_scope('summaries'):
        tf.summary.histogram('hist', var)

def run(inputs, targets):

    x = tf.placeholder(tf.float32, [None, 2], name='x')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y_')
    
    with tf.name_scope('layer0'):
        with tf.name_scope('weights'):
            w = tf.Variable(tf.random_normal([2,4]))
            var_summary(w)
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([4]))
            var_summary(b)
        with tf.name_scope('layer_out'):
            mid = tf.nn.relu(tf.add(tf.matmul(x, w), b))
            var_summary(mid)

    with tf.name_scope('layer1'):
        with tf.name_scope('weights'):
            w1 = tf.Variable(tf.random_normal([4,2]))
            var_summary(w1)
        with tf.name_scope('bias'):
            b1 = tf.Variable(tf.zeros([2]))
            var_summary(b1)
        with tf.name_scope('layer_out'):
            out = tf.nn.softmax(tf.add(tf.matmul(mid, w1), b1))
            var_summary(out)
        
    
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out)
    cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    #loss = tf.losses.mean_squared_error(labels=y_, predictions=out)
    #tf.summary.scalar('loss', loss)

    with tf.name_scope('training'):
        #train_op = tf.train.AdamOptimizer(1.5).minimize(loss)
        train_op = tf.train.AdamOptimizer(.05).minimize(cross_entropy)

    with tf.name_scope('testing'):
        with tf.name_scope('correct'):
            correct = tf.equal(out, y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    sess = tf.InteractiveSession()

    merged = tf.summary.merge_all()
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter("summary", sess.graph)
    test_writer = tf.summary.FileWriter("summary")
    train_writer.add_graph(sess.graph)
    test_writer.add_graph(sess.graph)

    batches = []
    batchsize=10
    for i in range(int(len(inputs)/batchsize)):
        inp = inputs[i*batchsize:(i+1)*batchsize]
        tar = targets[i*batchsize:(i+1)*batchsize]
        batches.append([inp, tar])
        
    for i in range(100):
        for j in range(len(batches)):
            summary, _ = sess.run([merged, train_op], feed_dict={x:batches[j][0], y_:batches[j][1]})
            train_writer.add_summary(summary, i*1000 + j)
        weights, summary, acc = sess.run([w1, merged, accuracy], feed_dict={x:inputs, y_:targets})
        test_writer.add_summary(summary, i*10000)
        print(acc)
        print(weights)


        result = sess.run([out], feed_dict={x:[[0,0]], y_:[[0,1]]})
        print(result)
        result = sess.run([out], feed_dict={x:[[0,1]], y_:[[1,0]]})
        print(result)
        
