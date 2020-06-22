#!/usr/bin/env python
# coding: utf-8

# In[164]:


from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# In[165]:


print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))


# In[166]:


import numpy as np
X_train=np.reshape(X_train,(60000,-1))
X_test=np.reshape(X_test,(10000,-1))
a=np.zeros((60000,10))
for i in range(0,60000):
    a[i][Y_train[i]]=1
Y_train=a
a=np.zeros((10000,10))
for i in range(0,10000):
    a[i][Y_test[i]]=1
Y_test=a
print(Y_test)


# In[174]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
n_input=784
n_hidden1=512
n_hidden2=256
n_hidden3=128
n_output=10
learning_rate=1e-4
n_iterations=1000
batch_size=60
dropout=0.8
X=tf.placeholder("float",[None,n_input])
Y=tf.placeholder("float",[None,n_output])
keep_prob=tf.placeholder(tf.float32)
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']


# In[175]:


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
        ))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[176]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
p=0
for i in range(n_iterations-1):
    batch_x, batch_y = X_train[p:p+60],Y_train[p:p+60]
    p=p+60
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })

    # print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy)
            )

test_accuracy = sess.run(accuracy, feed_dict={X: X_test, Y:Y_test , keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)


# In[ ]:





# In[ ]:




