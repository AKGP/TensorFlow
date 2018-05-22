import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# 28x28 px 
x = tf.placeholder(tf.float32,shape=[None,784])

#y_ is y bar
y_= tf.placeholder(tf.float32,[None,10  ])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# calculated y with error
y = tf.nn.softmax(tf.matmul(x,W) + b)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)


# batch_xs is image data and batch_ys is the true digit for image
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
test_accuracy = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
print("Test Accuracy of {0}%".format(test_accuracy * 100.0))
sess.close()





