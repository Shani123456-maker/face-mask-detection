#!/usr/bin/env python
# coding: utf-8



# In[ ]:


# this file will get data of images through the webcam and saves it to files
import cv2 # open cv library for loading image and converting it to arrays
import numpy as np  # numerical python library used to handle n dimensional arrays
from sklearn.svm import SVC # to train the model by giving the image data
from sklearn.metrices import accuracy_score # checks the accuracy of model after testing
from sklearn.model_selection import train_test_split # splits the training and testing data


# In[ ]:


image = cv2.imread('Downloads/R.jpg') # imports dummy image to see the dimensions


# In[ ]:


image.shape # image's dimensions are given that are with height and the color which is rgb


# In[ ]:


image[0]  # displays the pixels at zero first row


# In[ ]:


image # displays whole inage in the  form of arrays


# In[ ]:


while True:
    cv2.imshow('result',image)
    if cv2.waitKey(2) == 27:
        break
        
cv2.destroyAllWindows()


# In[ ]:


# haar features are the features that are common for all humans, there is an xml that contains the features of human face
# i.e the middle portion of nose is light and other is dark, 
haar_data = cv2.CascadeClassifier('Documents/data.xml') # imports file having data haar features


# In[ ]:


haar_data.detectMultiScale(image)  # returns x,y,width,height of the image


# In[ ]:


# this sectiion checks for haar features in the image and collect the face dimensions. It displays the image in a new window 
# that is closed when the escape key is pressed.

while True:
    face = haar_data.detectMultiScale(image) # detects haar features
    for x,y,w,h in face: # here the face is being detected 
        cv2.rectangle(image,(x,y),(x+w , y+h), (255,0,255), 4)  # makes the rectangle around image
    cv2.imshow('result',image) # shows image in new window
    if cv2.waitKey(2) == 27: # 27 is assci value of esc . window exits when esc key is pressed
        break
        
cv2.destroyAllWindows()


# In[ ]:


# in this section the images are being captured from the webcame. 400 images can be captured out of
# which 200 are with mask and 200 with out mask.

# at first images are captured and saved in a with_mask.npy file and the other half is capture and saved in 
# without_mask.npy
capture = cv2.VideoCapture(0) # gets the camera of laptop open
data= [] # list for data of faces
while True:
    flag, image = capture.read() #image is the image captured through the camera while flag is the variable that shows whether the camera is working or not
    if flag:
        face = haar_data.detectMultiScale(image)   #  detects haar features
        for x,y,w,h in face:
            cv2.rectangle(image,(x,y),(x+w , y+h), (255,0,255), 4)   # makes the rectangle around image
            face = image[y:y+h, x:x+w, :] # slices the face area, rows, collumns , color
            cv2.resize(face, (50,59)) # convert images to a same dimension
            print (len(data)) # prints the face number captured
            if len(data) < 400: # collects data upto 400 faces
                data.append(face) # appends the faces data in the list
        cv2.imshow('result',image) # shows image in new window
        if cv2.waitKey(2) == 27 or len(data) >= 200:  # 27 is assci value of esc . window exits when esc key is pressed also it collects data for 200 faces and stops the capture
            break
capture.release()        
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:


np.save('with_mask.npy',data) # saves images with mask data to a file


# In[ ]:


np.save('without_mask.npy',data) # save images without mask data to a file




