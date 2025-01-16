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
# Load the trained model
model = load_model('Eye_Disease_Detection.keras')

# Define class labels (you can change them to match your specific dataset)
CLASS_NAMES = ['cataract', 'diabetic_retinopathy','glaucoma', 'normal']
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess the image and predict the class
def prepare_image(img_bytes):
    img = Image.open(BytesIO(img_bytes)).resize((256, 256))  # Resize image to match input size
    img_array = np.array(img) # Normalize the image
    img_array=preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Check file type using imghdr
        file_type = imghdr.what(file)
        if file_type not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'Invalid file type. Please upload a valid image.'})
        
    # Save the uploaded file temporarily
    img_bytes=file.read()
    
    # Preprocess the image and predict
    img_array = prepare_image(img_bytes)
    predictions = model.predict(img_array)
    print(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    predicted_prob = float(np.max(predictions))
    

    return jsonify({'prediction': predicted_class, 'confidence': predictions[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True)
