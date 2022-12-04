# Indian-Sign-Language-Recognition

## Introduction 

Sign language becomes a primary form of communication when communicating with people with deaf/mute condition. And moreover many people don't know about sign languages since it's not part of any academic curriculum. It is hence a vital part of communication in society. We used a 
This project helps in predicting the right characters for a given set of images. 

## Aim: 
Detect the right character for a given image


## Dataset: Indian Sign Language (ISL)
ISL contains 128x128 pixel pictures of gestures belonging to characters A-Z.The labels/characters are stored as the folders with it's images. Dataset link - https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset?select=data

## Model Used : CNN
Convolutional Nueral Network(CNN) works best for image classification. Its built-in convolutional layer reduces the high dimensionality of images without losing its information. CNN also contains a pooling layer that not only reduces the dimensionality but also helps overcome transitional invariance. This is one of the reasons why we choose CNN over ANN. This model does'nt require any preprocessing to be done for the images.This model is more advantageous than model built as feature extraction and classifying using k-means clustering/SVM.

We make use of 2 Convolutional layers,max pool layers and batch normalization with drop outs. 

Image of model used -

![my_model](https://user-images.githubusercontent.com/118037785/205486370-268cf54f-00d4-4f96-9f58-62ff126797da.png)

## Result 
Confusion matrix -


![cm](https://user-images.githubusercontent.com/118037785/205486479-43136b08-358d-4468-8544-7e7063209898.png)

Accuracy: 0.9997652214744092
Precision_score:  0.999766446405847  
F1 score: 0.9997652236010504  
Recall score: 0.9997652214744092

## Conclusion
To conclude, sign language recognition can easily be processsed by cnn our model.This model as seen above as highest accuracy.
