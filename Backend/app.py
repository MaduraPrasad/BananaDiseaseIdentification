from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Establish connection to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB connection string
db = client['bananaDisease']  # Update with your database name
newPrediction_collection = db['newPrediction']  # Collection for diseases information
treatment_collection = db['treatment']  # Collection for treatment information
control_method_collection = db['controlMethods']  # Collection for control methods information
diseaseInfo_collection = db['diseaseInfo']


# Load the pre-trained model
model = tf.keras.models.load_model('my_model.h5')

# Define allowed image extensions and upload folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Map class indices to class names
class_names = {0: 'Cordana', 1: 'Healthy', 2: 'Panama', 3: 'Yellow Sigatoka'}  # Modify this dictionary with your class names

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255
        # Predict class probabilities
        preds = model.predict(x)
        # Get the predicted class index
        predicted_class = np.argmax(preds[0])
        # Get the predicted class name from the dictionary
        predicted_class_name = class_names[predicted_class]
        
        # Insert prediction into MongoDB (diseasesInfo collection)
         # Insert prediction details into MongoDB (newPrediction collection)
        newPrediction_collection.insert_one({
            'filename': filename,
            'prediction': predicted_class_name,
            'probabilities': {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
        })
        
        return jsonify({
            'prediction': predicted_class_name
        })
        
        #return jsonify({'prediction': predicted_class_name})
    else:
        return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
