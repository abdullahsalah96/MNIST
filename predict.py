from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import cv2





###################IMAGE###############################

model = load_model('model.h5')
img = cv2.imread('9.jpg')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.resize(grey, (28,28)) #resize the image to be the match the input of our model
ret,thresh = cv2.threshold(img2,20,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret,thresh2 = cv2.threshold(grey,20,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
img2 = np.array(thresh)
img2 = img2.astype('float32')/255 #normalizing the image
# print('img matrix: ', img)
img3 = img2.reshape(28, 28, 1)
img3 = np.expand_dims(img3, axis=0)
prediction = model.predict(img3) #Get the prediction array of the model after reshaping the image
# print('Prediction matrix: ', prediction)
prediction = np.argmax(prediction)  #get the index where the probability is maximum
cx, cy = img.shape[1]/2, img.shape[0]/2
print('Predicted number: ', prediction)
cv2.putText(img, str(prediction), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 0, 0), lineType=cv2.LINE_AA)
cv2.imshow('img', img)
cv2.imshow('thresh', thresh2)
while(cv2.waitKey(1)!=27):
    cv2.imshow('img', img)
    cv2.imshow('thresh', thresh2)

#########################VIDEO#########################

# cap = cv2.VideoCapture()
# cap.open(0)
# while True:
#     ret, img = cap.read()
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.resize(img, (28,28)) #resize the image to be the match the input of our model
#     # img = cv2.imread('D:/Courses/Udacity Nanodegrees/Machine Learning Nanodegree/self_projects/Text recognition using keras/1.png', cv2.IMREAD_GRAYSCALE) #load image as grayscale
#     # img2 = cv2.bitwise_not(img) #invert the image to match the mnist dataset
#     ret,thresh = cv2.threshold(img2,20,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#     ret,thresh2 = cv2.threshold(img,20,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
#     img2 = np.array(thresh)
#
#     img2 = img2.astype('float32')/255 #normalizing the image
#     # print('img matrix: ', img)
#     img3 = img2.reshape(28, 28, 1)
#     img3 = np.expand_dims(img3, axis=0)
#     prediction = model.predict(img3) #Get the prediction array of the model after reshaping the image
#     # print('Prediction matrix: ', prediction)
#     prediction = np.argmax(prediction)  #get the index where the probability is maximum
#     print('Predicted number: ', prediction)
#     cv2.putText(img, str(prediction), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
#     cv2.imshow('img', img)
#     cv2.imshow('thresh', thresh2)
#     if (cv2.waitKey(1)==27):
#         break
