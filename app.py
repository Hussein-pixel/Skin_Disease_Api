from flask import Flask, request, jsonify
import numpy as np
import keras
from keras.preprocessing import image
import pandas as pd
import tensorflow as tf
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import cv2
import os
from read_img import classify_image



# Create a Flask app
app = Flask(__name__)
# limiter = Limiter(get_remote_address, app=app)

# Dirs & Extensions
UPLOAD_DIRECTORY = "uploads"
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg"]

def allowed_file(file_name):
    return "." in file_name and file_name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    html = '''
    <html>
        <head>
            <title>Hello</title>
        </head>
        <body>
            <h1>Hello Skin Disease Model</h1>
        </body>
    </html>
    '''
    return html

# Define a route for the API
@app.route('/classify', methods=['POST'])
# @limiter.limit("10/minute")  # Limit to 10 requests per minute.
def predict_class():
    try:
        # Get the image file from the request
        file = request.files['img']
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400
        elif file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
            file.save(file_path)      
            
            # read image and return results
            result = classify_image(file_name)
            return result
        else:
            return jsonify({"error": "Invalid file, We can only accept JPG,JPEG or PNG images"}), 400 
    except KeyError:
        return jsonify({"error": "Missing img parameter"}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=80)