#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#to read path of the dataset images

path = os.getcwd()+"\\Indian\\" 
folder = []

for i in os.listdir(path):
    folder.append(path+i)

folder = folder[9:]
folder


# In[2]:


t={}

# to store the labels for future mapping
for i in range(26):
    t[i] = chr(ord('A')+int(i))
    
t


# In[3]:


# import cv2
# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


# In[4]:


import cv2
from tqdm import tqdm

features = []
labels=[]

#to read 50 x 50 sized images as features and their corresponding labels

for i in folder:
#     count = 0
    for img in tqdm(os.listdir(i)):
        f = cv2.imread(os.path.join(i,img))
        f = cv2.cvtColor(f , cv2.COLOR_BGR2GRAY)
        f = cv2.resize(f , (50,50))
#         if count == 0:
#             key1 = sift.detect(f,None)
        features.append(f)
#         else : 
#             key2 = sift.detect(f,None)
#             features.append(bf.match(desp1,desp2))
#         count += 1
        labels.append(ord(i.replace(path, ""))-ord('A'))
        


# In[5]:


labels


# In[6]:


import numpy as np
#converting features and labels to numpy array for computation
X = np.array(features)
print(X.shape)

Y = np.array(labels)
print(Y.shape)
Y


# In[7]:


import keras
from keras.utils import np_utils
#reshaping the images for computation

X = X.reshape(31945,50,50)
X = np_utils.normalize(X)

Y = np_utils.to_categorical(Y)
Y


# In[8]:


from sklearn.model_selection import train_test_split

#splitting the dataset into training and testing sets
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.6,test_size = 0.4, random_state = 42, stratify = Y)


# In[9]:


import tensorflow as tf
from keras import models
from keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

#cnn model
model = models.Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (50,50,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(layers.Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(layers.Dense(units = 26 , activation = 'softmax'))
# model.add(Dense(36, kernel_regularizer=tf.keras.regularizers.l2(0.001),activation='softmax'))
# model.compile(optimizer = 'adam', loss = 'squared_hinge', metrics = ['accuracy'])


# In[10]:


model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()


# In[11]:


# model1 = 


# In[12]:


xtrain=xtrain/255
xtest=xtest/255


# In[13]:


#fitting the training set and computing accuracy , loss at each epoch
history = model.fit(xtrain,ytrain ,epochs = 5 , validation_data = (xtest, ytest))


# In[14]:


#predicting the labels from testing feature set  
pred=model.predict(xtest)


# In[15]:


from sklearn import metrics
import matplotlib.pyplot as plt
import itertools

#computing confusion matrix and plotting it

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_pred=np.argmax(pred,axis=1)
y_test=np.argmax(ytest,axis=1)
cm = metrics.confusion_matrix(y_test, y_pred)
# print(cm)
# print('Accuracy' + str())
plot_confusion_matrix(cm, classes = list(t.values()))
plt.show()


# In[16]:


print(metrics.accuracy_score(y_test, y_pred))


# In[17]:


import sklearn.metrics as sm
#calculating the metrcs for valuation of the model

print("precision_score for ",sm.precision_score(y_test,y_pred,average='weighted'))
print("f1 score for ",sm.f1_score(y_test,y_pred,average='weighted'))
print("recall score for ",sm.recall_score(y_test,y_pred,average='weighted'))


# In[18]:


import pandas as pd

l = [t[i] for i in y_pred]
k = [t[i] for i in y_test]

final = pd.DataFrame(zip(l,k),columns=["Predicted letters","Actual letters"])


# In[19]:


#comparing the actual labels v/s the predicted labels
final


# In[32]:


#Multiclass classification 
from sklearn.metrics import classification_report
#generating classification report for further analysis of the prediction
for i in range(cm.shape[0]):
    TP = cm[0,0]
    FP = cm[0,:].sum() - tp
    FN = cm[:,0].sum() - tp
    TN = cm.sum().sum() - tp -fp -fn
    Accuracy = (TP+TN)/cm.sum().sum()
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1_Score = (2 * Precision *Recall)/(Precision+Recall)
#     print(t[i],Accuracy,Precision,Recall,F1_Score)

p = pd.DataFrame(classification_report(l,k,output_dict=True))


# In[31]:


p_transp = p.T
p_transp


# In[ ]:




