import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

loaded_model = tf.keras.models.load_model("digitsCNN.h5")

loaded_model.summary()

#passes img from camera to predict digit
def predict(model, img):
    imgs = np.array([img])
    result = model.predict(imgs)
    index = np.argmax(res)
    print(index)
    return str(index)

#opens camera
cap = cv2.VideoCapture(0)
