import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


WIDTH = 640
HEIGHT = 480

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

cap.set(3,WIDTH)
cap.set(4,HEIGHT)

#threshold 
#threshold = 100

def get_img_contour_thresh(img):
    """method to get back three elements, original img, contours detect
    and threshold image"""
    x, y, w, h = 0, 0, 300, 300
    
    # Change color-space from RGB -> Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur and Threshold
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Making thresh image of size x, y, w, h = 0, 0, 300, 300
    thresh = thresh[y:y+h,x:x+w]
    
    # Find contours
    contours, _= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    return img, contours, thresh


while True:
    prediction = ' '
    sucess, img = cap.read()
    cv2.imshow("name", img)

    #receives img, contours and applies a thershold to the camera
    img, contours, thresh= get_img_contour_thresh(img)
    
    cv2.imshow("thresh", thresh)
    #if something is detected
    if len(contours) > 0:
        #find the contour with the maximum area
        contour = max(contours, key = cv2.contourArea)
        #makes sure contour area is within certain range
        #above 1500 pixels but below 5000 pixels
        if cv2.contourArea(contour) > 1500 and cv2.contourArea(contour) < 5000:
            #creates a bounding box around each contour and returns the x, y, w, h of the rectangle
            x, y, w, h = cv2.boundingRect(contour)

            #images are just arrays so just making an updating image
            #this image will be used to predict the contours
            testImg = thresh[y:y+h, x:x+w]
            cv2.imshow("testIMG", testImg)
            #resize image to a size for our CNN
            testImg = cv2.resize(testImg, (28, 28))
            testImg = np.array(testImg)

            #Applies Histogram of Oriented Graident, a feature descriptor
            #hog_ft = hog(newImage, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            #hog_ft = 




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break