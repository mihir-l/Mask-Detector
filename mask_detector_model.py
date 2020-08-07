import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import cv2
from tensorflow.keras.callbacks import TensorBoard

NAME = "maskdetector2DConv-2"


X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))


#X_train = tf.keras.utils.normalize(X_train, axis=1)
X_train = np.array(X_train/255.0)
y_train = np.array(y_train)

X_test = np.array(X_test/255.0)
y_test = np.array(y_test)


##X = pickle.load(open("X.pickle", "rb"))
##y = pickle.load(open("y.pickle", "rb"))
##
##X = np.array(X/255.0)
##y = np.array(y)

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X_train.shape[1:]))
model.add(Activation("relu"))#rectified linear func
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))#rectified linear func
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))          

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=25, validation_split=0.1, callbacks=tensorboard)

model.save('mask_detector_model')

##val_loss, val_acc = model.evaluate(X_test, y_test)
##print(val_loss, val_acc)

