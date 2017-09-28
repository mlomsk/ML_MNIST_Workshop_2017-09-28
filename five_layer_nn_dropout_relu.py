import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math


def nn_layer(X, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
        with tf.name_scope('biases'):
            B = tf.Variable(tf.zeros(output_dim))
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(X, W) + B
            Y = act(preactivate)

        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', B)
        tf.summary.histogram('pre_activations', preactivate)
        tf.summary.histogram('activations', Y)
        
        return Y, preactivate

def model(mnist, epoches=1000, batch_size=100, learning_rate=0.003):
    print("Start model")
    layer_sizes = [28*28, 200, 100, 60, 30, 10]

    with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, [None, 784], name='X')
        x_image = tf.reshape(X, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 10)

    Y1 = nn_layer(X, layer_sizes[0], layer_sizes[1], "first")[0]
    Y2 = nn_layer(Y1, layer_sizes[1], layer_sizes[2], "second")[0]
    Y3 = nn_layer(Y2, layer_sizes[2], layer_sizes[3], "third")[0]
    Y4 = nn_layer(Y3, layer_sizes[3], layer_sizes[4], "fourth")[0]
    Y,Ylogits = nn_layer(Y4, layer_sizes[4], layer_sizes[5], 'fifth', tf.nn.softmax)

    with tf.name_scope('Y_'):
        Y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('xentropy'):
        # Функция потерь H = Sum(Y_ * log(Y))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
        cross_entropy = tf.reduce_mean(cross_entropy)*100
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # Доля верных ответов найденных в наборе
            is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
        with tf.name_scope('xentropy_mean'):
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('train'):
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000.0
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # Оптимизируем функцию потерь меотодом градиентного спуска
        # 0.003 - это шаг градиента, гиперпараметр
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Минимизируем потери
        train_step = optimizer.minimize(cross_entropy)

    with tf.Session() as sess:
        merged = tf.summary.merge_all() # Merge all the summaries and write them out to
        writer = tf.summary.FileWriter("/tmp/tensorflow/five_layer_nn_dropout_relu", sess.graph)
        tf.global_variables_initializer().run()

        for i in range(epoches):
            # загружаем набор изображений и меток классов
            batch_X, batch_Y = mnist.train.next_batch(batch_size)

            lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
            train_data={X: batch_X, Y_: batch_Y, learning_rate: lr}

            # train
            sess.run(train_step, feed_dict=train_data)
            
            if i % 10 == 0:
                test_data={X: mnist.test.images, Y_: mnist.test.labels}
                summary, a = sess.run([merged, accuracy], feed_dict=test_data)
                writer.add_summary(summary, i)
                if i % 200 == 0:
                    print("Test: {}".format(a))
            
        writer.close()

def main():
    print("MNIST single layer NN")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=True)
    tf.set_random_seed(0)
    tf.reset_default_graph()
    model(mnist, epoches=10000)

if __name__ == '__main__':
    main()
