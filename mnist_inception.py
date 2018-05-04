from keras.datasets import mnist
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras import backend as K
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def one_hot(vec,num_classes=10):
    m = vec.shape[0]
    oh_vec = np.zeros((m,num_classes))
    for i in range(0,m):
        oh_vec[i,vec[i]] = 1
    return oh_vec

def inception_block(X, layers):
    row1 = layers.Conv2D(layers[0],(1,1),padding="same",activation="relu")(X)
    row2 = layers.Conv2D(layers[1],(1,1),padding="same",activation="relu")(X)
    row2 = layers.Conv2D(layers[1],(3,3),padding="same",activation="relu")(row2)
    row3 = layers.Conv2D(layers[2],(1,1),padding="same",activation="relu")(X)
    row3 = layers.Conv2D(layers[2],(5,5),padding="same",activation="relu")(row3)
    row4 = layers.MaxPooling2D((3,3),strides=(1,1))(X)
    row4 = layers.Conv2D(X._keras_shape[2],(1,1),padding="same",activation="relu")(row4)
    result = layers.Concatenate([row1,row2,row3,row4])
    return result

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
perm = np.random.permutation(train_data.shape[0])
train_data = train_data[perm,:,:]
train_labels=train_labels[perm]
num_train = 54000
x_train = train_data[:num_train,:,:]
x_train = x_train.reshape((x_train.shape[0],28,28,1))
x_validate = train_data[num_train:,:,:]
x_validate = x_validate.reshape((x_validate.shape[0],28,28,1))
y_train = one_hot(train_labels[:num_train])
y_validate = one_hot(train_labels[num_train:])

x_test = test_data
x_test = x_test.reshape((x_test.shape[0],28,28,1))
y_test = one_hot(test_labels)

X = Input(shape=(28,28,1))
X = MaxPooling2D((2,2),strides=(2,2))(X)
X = inception_block(X,[32,32,32])
