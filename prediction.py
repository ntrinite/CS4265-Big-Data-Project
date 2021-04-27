import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics

from keras.models import load_model

#images will be scaled to 28x28
HEIGHT = 28
WIDTH = 28

#loading test and trainig data
train = pd.read_csv("emnist-balanced-train\emnist-balanced-train.csv",delimiter = ',')
test = pd.read_csv("emnist-balanced-test\emnist-balanced-test.csv",delimiter = ',')

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


#mapping so values can come out as what they actually are
label_map = pd.read_csv("emnist-balanced-mapping.txt", 
                        delimiter = ' ', 
                        index_col=0, 
                        header=None, 
                        squeeze=True)


#output is represented as an int. This helps with mapping
label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)

print(label_dictionary)




















WIDTH = 640
HEIGHT = 480

model = load_model("digitsCNN.h5")

y_pred = model.predict(X_val)
for i in range(42, 48):
    plt.subplot(380 + (i%10+1))
    plt.imshow(X_val[i].reshape(28, 28), cmap=plt.cm.gray)
    plt.title(label_dictionary[y_pred[i].argmax()])
plt.show()
model.summary()

#opens camera
cap = cv2.VideoCapture(0)

cap.set(3,WIDTH)
cap.set(4,HEIGHT)

while True:
    prediction = ' '
    sucess, img = cap.read()

    #converts image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)

    #applys gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    #applies threshold, receives binary iamge
    _, binImg = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow("bin", binImg)

    resized = cv2.resize(binImg,(28,28))
    #print("reized ", resized.shape)

    #invert binary image
    imgInvert = 255 - resized
    cv2.imshow("invert", resized)

    #resize img for model
    modelImg = resized.reshape(1,28,28,1)

    prediction = model.predict(modelImg)
    prediction = np.argmax(prediction,axis=1)[0]
    cv2.putText(img,'Predicted Digit : '+label_dictionary[prediction],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(69,255,69),1)  
    print(label_dictionary[prediction])

    cv2.imshow("name", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
        
