import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

#loading the data from the keras librarires
(X_train,Y_train), (X_test,Y_test) = cifar10.load_data()

#reshaping the data so thta it can be used by keras' conv layers
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')

#normalizing the data so it ranges between 0 and 1
X_train /= 255

X_test /= 255

#creating one-hot encoding of output labels from dataset
number_of_classes = 10
Y_train = np_utils.to_categorical(Y_train, number_of_classes)
Y_test = np_utils.to_categorical(Y_test, number_of_classes)

model = Sequential()
model.add(Conv2D(32, 5, input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
model.add(Conv2D(32, 5, activation='relu'))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, 5, activation='relu'))
model.add(Conv2D(32, 5, activation='relu'))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, 5, activation='relu'))
model.add(Conv2D(32, 5, activation='relu'))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 2, batch_size = 128)
model.fit(X_train, Y_train, epochs = 1, batch_size = 256)

model.save('CIFAR10_1.h5')

scores = model.evaluate(X_test, Y_test, batch_size=32)
print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
