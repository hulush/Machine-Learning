import tensorflow as tf
import tensorflow.keras as keras
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train= tf.keras.utils.normalize(x_train,axis=1)    # train
x_test= tf.keras.utils.normalize(x_test,axis=1)      # test
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))    #Final layer. It has 10 nodes. 1 node per possible number prediction.
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)    # layer

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

import matplotlib.pyplot as plt
print(x_train[0])
plt.imshow(x_train[0],cmap= plt.cm.binary)   # color binary
plt.show()

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict(x_test)
print(predictions)
import numpy as np
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()
