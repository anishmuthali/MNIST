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

inputs = layers.Input(shape=(28,28,1))
X = inputs
X_res = X
X = layers.Conv2D(16,(3,3),strides=(1,1),padding="same",activation="relu")(X)
X = layers.Conv2D(32,(3,3),strides=(1,1),padding="same",activation="relu")(X)
X = layers.Conv2D(16,(3,3),strides=(1,1),padding="same",activation="relu")(X)
added = layers.Add()([X,X_res])
X = layers.Activation(activation="relu")(added)
X = layers.Conv2D(16,(3,3),strides=(1,1),padding="valid",activation="relu")(X)
X = layers.Conv2D(32,(3,3),strides=(1,1),padding="valid",activation="relu")(X)
X = layers.MaxPooling2D((2,2),strides=2)(X)
X = layers.Conv2D(16,(3,3),strides=(1,1),padding="valid",activation="relu")(X)
X = layers.MaxPooling2D((2,2),strides=2)(X)
X = layers.Flatten()(X)
X = layers.Dense(128,activation="relu")(X)
X = layers.Dense(64,activation="relu")(X)
outputs = layers.Dense(10,activation="softmax")(X)


epochs=20
model = models.Model(inputs=inputs,outputs=outputs)
optimizer = optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=512,epochs=epochs,verbose=2,validation_data=(x_validate,y_validate))
print("Finished fitting.")
epochs = range(1, epochs + 1)
hist_dict = history.history
plt.title("Accuracy vs Epochs")
plt.plot(epochs, hist_dict["acc"], 'bo', label="Training")
plt.plot(epochs, hist_dict["val_acc"], 'go', label="Validation")
plt.legend(loc="best")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

print("Checking accuracy on test set...")
acc = model.evaluate(x_test, y_test, batch_size=512)
print("Accuracy on test set: " + str(acc[1]))
plt.show()