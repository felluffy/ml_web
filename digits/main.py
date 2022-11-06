import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 
import cv2

(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()

x_tr = x_tr / 255. 
x_te = x_te / 255.

x_tr = x_tr.reshape(-1, 28, 28, 1)
x_te = x_te.reshape(-1, 28, 28, 1)

nums = np.unique(y_tr)
y_tr = keras.utils.to_categorical(y_tr, len(nums))
y_te = keras.utils.to_categorical(y_te, len(nums))

inp = keras.layers.Input(shape=(28,28,1))
c1_1 = keras.layers.Conv2D(32, 5, 1, 'SAME', activation='gelu')(inp)
c1_2 = keras.layers.Conv2D(32, 3, 1, 'same', activation='gelu')(c1_1)
b1 = keras.layers.BatchNormalization()(c1_2)
m1 = keras.layers.MaxPool2D()(b1)

c2_1 = keras.layers.Conv2D(32, 5, 1, 'same', activation='gelu')(m1)
c2_2 = keras.layers.Conv2D(32, 3, 1, 'same', activation='gelu')(c2_1)
b2 = keras.layers.BatchNormalization()(c2_2)
m2 = keras.layers.MaxPool2D()(b2)


c3_1 = keras.layers.Conv2D(64, 5, 1, 'same', activation='gelu')(m2)
c3_2 = keras.layers.Conv2D(64, 3, 1, 'same', activation='gelu')(c3_1)
b3 = keras.layers.BatchNormalization()(c3_2)
m3 = keras.layers.MaxPool2D()(b3)

fl = keras.layers.Flatten()(m3)
de = keras.layers.Dense(len(nums), 'softmax')(fl)

model = keras.Model(inputs=inp, outputs=de)
model.compile('nadam', 'categorical_crossentropy', 'accuracy')

model.fit(x_tr, y_tr, 256, 10, validation_data=(x_te, y_te))

model.save('mnist_model.hdf5')
