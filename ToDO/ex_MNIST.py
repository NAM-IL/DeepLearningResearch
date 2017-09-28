import tensorflow as tf
import numpy as np
import Network
# from tensorflow.examples.tutorials.mnist import input_data
import mnist_loader

# tf.set_random_seed(777)  # reproducibility

# mnist = input_data.read_data_sets("./data/", one_hot=True)

  
training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()


# print("training_data:{0}".format(training_data))
# print("validation_data:{0}".format(validation_data))
# print("test_data:{0}".format(test_data))
# print("validation_data:{0}".format(mnist.validation.num_examples))
 
net = Network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 100.0, test_data=test_data)
net.SGD(training_data, 70, 10, 3.0, test_data=test_data)