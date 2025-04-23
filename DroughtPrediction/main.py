# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

app = Flask(__name__, static_folder='static')

# Global variable to store the model
model = None

def load_model():
    """Load the trained DCNN model"""
    global model
    
    # Define the same model architecture as in your script
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    
    # Load weights if available
    if os.path.exists("model/model_weights.hdf5"):
        model.load_weights("model/model_weights.hdf5")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model loaded successfully")
    else:
        print("Warning: model weights file not found. Predictions will not be accurate.")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template("index.html")

@app.route('/team')
def team():
    """Serve the team page"""
    return render_template("team.html")

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to handle image prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # Save the file temporarily
    temp_path = 'temp_upload.jpg'
    file.save(temp_path)
    
    try:
        # Process the image similar to your script
        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Convert to BGRA (4 channels) as in your model
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Resize to 28x28 as expected by the model
        img = cv2.resize(image, (28, 28))
        
        # Reshape and normalize
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1, 28, 28, 4)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        
        # Make prediction
        if model is not None:
            preds = model.predict(img)
            predict_class = np.argmax(preds)
            
            # Determine if drought detected (classes 0 or 1 as in your script)
            is_drought = predict_class in [0, 1]
            
            return jsonify({
                'prediction': 'Drought Detected' if is_drought else 'No Drought Detected',
                'class': int(predict_class),
                'confidence': float(preds[0][predict_class])
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Load the model before starting the app
    load_model()
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Create uploads directory for temporary storage
    os.makedirs('static/uploads', exist_ok=True)
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)