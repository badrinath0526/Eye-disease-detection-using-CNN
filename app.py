from flask import Flask, render_template, request, jsonify
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
from PIL import Image
from io import BytesIO
app = Flask(__name__)
import imghdr
import logging

logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)

# Load the trained model
try:
    model = load_model('Eye_Disease_Detection.keras')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading the model:{str(e)}")

# Define class labels (you can change them to match your specific dataset)
CLASS_NAMES = ['cataract', 'diabetic_retinopathy','glaucoma', 'normal']
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess the image and predict the class
def prepare_image(img_bytes):
    try:
        logger.debug("Preparing image for prediction")
        img = Image.open(BytesIO(img_bytes)).resize((256, 256))  # Resize image to match input size
        img_array = np.array(img) # Normalize the image
        logger.debug(f"Image shape before preprocessing: {img_array.shape}")
        img_array=preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        logger.debug(f"Image shape after preprocessing: {img_array.shape}")
        return img_array
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'})
    
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No selected file'})
    
        if file and allowed_file(file.filename):
        # Check file type using imghdr
            file_type = imghdr.what(file)
            if file_type not in ALLOWED_EXTENSIONS:
                return jsonify({'error': 'Invalid file type. Please upload a valid image.'})
        
    # Save the uploaded file temporarily
        img_bytes=file.read()
        logger.debug(f"Received file with size {len(img_bytes)} bytes")

    
    # Preprocess the image and predict
        img_array = prepare_image(img_bytes)
        if img_array is None:
            return jsonify({'error':"Error processing the image"}),400
        predictions = model.predict(img_array)
        logger.debug(f"Model predictions: {predictions}")
    # print(predictions)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        predicted_prob = float(np.max(predictions))
    
        logger.info(f"Predicted class: {predicted_class} with confidence {predicted_prob*100}%")

        return jsonify({'prediction': predicted_class, 'confidence': predictions[0].tolist()})
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'Error': 'An error occured during prediction'})
if __name__ == '__main__':
    app.run(debug=True)
