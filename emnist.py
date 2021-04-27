import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import cv2

#keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

import sklearn.metrics as metrics

#loading test and trainig data
train = pd.read_csv("emnist-balanced-train\emnist-balanced-train.csv",delimiter = ',')
test = pd.read_csv("emnist-balanced-test\emnist-balanced-test.csv",delimiter = ',')
print("Train shape {}, Test shape {}".format(train.shape, test.shape))

#mapping so values can come out as what they actually are
label_map = pd.read_csv("emnist-dataset\emnist-balanced-mapping.txt", 
                        delimiter = ' ', 
                        index_col=0, 
                        header=None, 
                        squeeze=True)

#output is represented as an int. This helps with mapping
label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)

print(label_dictionary)

#images will be scaled to 28x28
HEIGHT = 28
WIDTH = 28

#splitting training and testing data to xtrain/test and ytrain/test
X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]

X_test = test.iloc[:,1:]
y_test = test.iloc[:,0]
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# Flip and rotate image
X_train = np.asarray(X_train)
X_train = np.apply_along_axis(rotate, 1, X_train)
print ("X_train:",X_train.shape)

X_test = np.asarray(X_test)
X_test = np.apply_along_axis(rotate, 1, X_test)
print ("X_test:",X_test.shape)


# Normalise so now each pixel is between 0 and 1
X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255

plt.imshow(X_train[3])
plt.show()

# number of classes
num_classes = y_train.nunique()


# One hot encoding
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)

# Reshape image for CNN
X_train = X_train.reshape(-1, HEIGHT, WIDTH, 1)
X_test = X_test.reshape(-1, HEIGHT, WIDTH, 1)

# splits to train and val
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.10, random_state=69)

#model
model = Sequential()

model.add(Conv2D(filters=128, kernel_size=(5,5), padding = 'same', activation='relu',input_shape=(HEIGHT, WIDTH,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3) , padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(units=num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=512, verbose=1, validation_data=(X_val, y_val))


model.save('digitsCNN.h5')

# plot accuracy and loss
def plotgraph(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
# Accuracy curve
plotgraph(epochs, acc, val_acc)
# loss curve
plotgraph(epochs, loss, val_loss)
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

scores = model.evaluate(X_val, y_val)
print(scores)
for i in range(42, 48):
    plt.subplot(380 + (i%10+1))
    plt.imshow(X_test[i].reshape(28, 28), cmap=plt.cm.gray)
    plt.title(label_dictionary[y_pred[i].argmax()])
plt.show()

cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
