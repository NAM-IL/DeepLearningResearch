import tensorflow as tf
import numpy as np
import Network
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("./data/", one_hot=True)

# 
# training_data, validation_data, test_data = \
#     mnist_loader.load_data_wrapper()


print("test_data:{0}".format(mnist))
print("test_data:{0}".format(mnist.train.num_examples))

net = Network.Network([784, 30, 10])
net.SGD(mnist.train, 30, 10, 100.0, test_data=mnist.test)