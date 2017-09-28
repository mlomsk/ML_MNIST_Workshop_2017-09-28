import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def model(mnist, epoches=1000, batch_size=100, learning_rate=0.003):
    print("Start model")
    with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, [None, 784], name='X')
        x_image = tf.reshape(X, [-1, 28, 28, 1])

    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]), name='weights')

    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='biases')
        
    with tf.name_scope('Wx_plus_b'):
        # Модель Y = X.W + b
        Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b, name='labels')

        # Подстановка для корректных значений входных данных
    with tf.name_scope('Y_'):
        Y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('xentropy'):
        # Функция потерь H = Sum(Y_ * log(Y))
        cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # Доля верных ответов найденных в наборе
            is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
        with tf.name_scope('xentropy_mean'):
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    with tf.name_scope('train'):
        # Оптимизируем функцию потерь меотодом градиентного спуска
        # 0.003 - это шаг градиента, гиперпараметр
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Минимизируем потери
        train_step = optimizer.minimize(cross_entropy)

    tf.summary.image('input', x_image, 10)
    tf.summary.histogram('weights', W)
    tf.summary.histogram('biases', b)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.Session() as sess:
        merged = tf.summary.merge_all() # Merge all the summaries and write them out to
        writer = tf.summary.FileWriter("/tmp/tensorflow/one_layer_nn", sess.graph)
        tf.global_variables_initializer().run()

        for i in range(epoches):
            # загружаем набор изображений и меток классов
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            train_data={X: batch_X, Y_: batch_Y}

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
