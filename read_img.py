from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import BatchNormalization
import os
import pandas as pd
import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model('EfficientNetB3-skin disease-86.74.h5')

def classify_image(uploaded_image):
    image = cv2.imread(os.path.join("uploads",uploaded_image),cv2.IMREAD_REDUCED_COLOR_2)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))
    image = np.expand_dims(image, axis=0)

    # Make a prediction using the model
    predictions = np.squeeze(model.predict(image))
    predicted_class = np.argmax(predictions)

    # Load Class_Dict To Get The Class_Index Corresponding Name 
    classes = pd.read_csv('class_dict.csv')

    # Return the predicted class as a JSON response
    response = jsonify(
            {'predicted_class': classes['class'].iloc[predicted_class],
             'Accuracy':f"{round(np.max(predictions),2) * 100} % "
            }
        )
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response
