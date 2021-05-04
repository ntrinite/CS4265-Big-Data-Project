import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from keras.models import load_model

# Images will be scaled to 28x28 resolution
HEIGHT = 28
WIDTH = 28

# Live capture resolution is 640x480
CAP_WIDTH = 640
CAP_HEIGHT = 480

# Loading test and trainig data
train = pd.read_csv("emnist-balanced-train\emnist-balanced-train.csv",delimiter = ',')
test = pd.read_csv("emnist-balanced-test\emnist-balanced-test.csv",delimiter = ',')

# Splitting training and testing data to xtrain/test and ytrain/test
X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]

X_test = test.iloc[:,1:]
y_test = test.iloc[:,0]

#print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# Flip and rotate image
X_train = np.asarray(X_train)
X_train = np.apply_along_axis(rotate, 1, X_train)
#print ("X_train:",X_train.shape)

X_test = np.asarray(X_test)
X_test = np.apply_along_axis(rotate, 1, X_test)
#print ("X_test:",X_test.shape)

# Normalise so now each pixel is between 0 and 1
X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255

# Number of classes
num_classes = y_train.nunique()

# One hot encoding
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
#print("y_train: ", y_train.shape)
#print("y_test: ", y_test.shape)

# Reshape image for CNN
X_train = X_train.reshape(-1, HEIGHT, WIDTH, 1)
X_test = X_test.reshape(-1, HEIGHT, WIDTH, 1)

# Splits to train and val
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.10, random_state=69)

# Mapping so values can come out as what they actually are
label_map = pd.read_csv("emnist-dataset\emnist-balanced-mapping.txt", 
                        delimiter = ' ', 
                        index_col=0, 
                        header=None, 
                        squeeze=True)

# Output is represented as an int. This helps with mapping
label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)

print(label_dictionary)

# Load Model
model = load_model("CNN_model.h5")

scores = model.evaluate(X_val, y_val)
y_pred = model.predict(X_val)
f1 = metrics.f1_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted')
percision = metrics.precision_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted')
recall = metrics.recall_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted')
print("Accuracy: {}\n Loss: {} \n F1 Score: {} \n Percision: {} \n Recall: {}".format(scores[1],scores[0],f1,percision,recall))

# Opens camera
cap = cv2.VideoCapture(0)

cap.set(3,CAP_WIDTH)
cap.set(4,CAP_HEIGHT)

while True:
    prediction = ' '
    sucess, img = cap.read()

    # Converts image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)

    # Applys gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    # Applies threshold, receives binary iamge
    _, binImg = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow("bin", binImg)

    cv2.imshow("bin", binImg)

    resized = cv2.resize(binImg,(28,28))
    #print("reized ", resized.shape)

    # Resize img for model
    modelImg = resized.reshape(1,28,28,1)

    prediction = model.predict(modelImg)
    prediction = np.argmax(prediction,axis=1)[0]
    cv2.putText(img,'Predicted Digit : '+label_dictionary[prediction],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(69,255,69),1)  
    print(label_dictionary[prediction])

    cv2.imshow("name", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break