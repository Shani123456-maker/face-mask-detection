#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# this file will load data of files and will perform algorithm to train and test the data, and after that it will capture video through the webcame to tell whether the person has mask or not
import cv2 # open cv library for loading image and converting it to arrays
import numpy as np  # numerical python library used to handle n dimensional arrays
from sklearn.svm import SVC # to train the model by giving the image data
from sklearn.metrices import accuracy_score # checks the accuracy of model after testing
from sklearn.model_selection import train_test_split # splits the training and testing data


# In[ ]:


# loads data of faces/loads files
with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')


# In[ ]:


with_mask.shape # gives the dimensions of image is this format (number of images,width,height,color(rgb))
without_mask.shape


# In[ ]:


with_mask = with_mask.reshape(200,50*50*3) # here images are being converted to 2D
without_mask = without_mask.reshape(200,50*50*3)


# In[ ]:


X = np.r_[with_mask,without_mask] # concatenates data into single array , here first 200 indexes are of with mask and other 200 are without mask

X.shape #gives the single array having data of both mask and without mask images


lables = npzeros(X.shape[0]) # gives all indexes zero

labels[200:] = 1.0 # now our task is to assign first 200 indexes the value 0 and other 200 value 1. This line gives second half value 1

names = {0: 'Mask' , 1 : 'No Mask'} # gives values to the integers which are printing mask and non mask


x_train, xtest, ytrain, y_test = trains_test_splits(X, labels, test_size=0.25) # it will split the data out of which 300 images will be used to train model and 100 will test the data

y_pred = svm.predict(x_test) # predicts the out come 

accuracy_score(y_test,y_pred) # gives the accuracy of model


# In[ ]:


# in this section the model is already trained and is tested on live webcame
haar_data = cv2.CascadeClassifier('Documents/data.xml') # imports file having data haar features
capture = cv2.VideoCapture(0) # gets the camera of laptop open
data= [] # list for data of faces
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, image = capture.read() #image is the image captured through the camera while flag is the variable that shows whether the camera is working or not
    if flag:
        face = haar_data.detectMultiScale(image)   #  detects haar features
        for x,y,w,h in face:
            cv2.rectangle(image,(x,y),(x+w , y+h), (255,0,255), 4)   # makes the rectangle around image
            face = image[y:y+h, x:x+w, :] # slices the face area, rows, collumns , color
            cv2.resize(face, (50,50)) # convert images to a same dimension
            pred = svm.predict(face)[0] # it predicts that whether the face has mask or not
            n = names[int(pred)] # takes the integer value of pred i.e 0 or 1, and convert is the name value that is either with_mask or without_mask
            cv2.putText(img,n,(x,y),font,1,(244,222,242),2) # puts a text on the camera to show the mask or not mask
            print(n) # prints the value of name on the camera
        cv2.imshow('result',image) # shows image in new window
        if cv2.waitKey(2) == 27 or len(data) >= 200:  # 27 is assci value of esc . window exits when esc key is pressed also it collects data for 200 faces and stops the capture
            break
capture.release()        
cv2.destroyAllWindows()

