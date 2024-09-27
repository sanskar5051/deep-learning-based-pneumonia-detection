import os
import sys
import io
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Ensure proper stdout encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the VGG19 model without the top layer
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))  # Use 224x224 input shape
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)

# Create the model
model_03 = Model(base_model.inputs, output)

# Load weights (make sure the weights match this architecture)
model_03.load_weights('vgg_unfrozen.h5')

# Initialize the Flask app
app = Flask(__name__)

# Print statement to confirm the model is loaded
print('Model loaded. Check http://127.0.0.1:5000/')

# Function to map prediction results to class names
def get_className(classNo):
    if classNo == 0:
        return "Normal"
    elif classNo == 1:
        return "Pneumonia"

# Function to preprocess the image and get the prediction
def getResult(img_path):
    # Read and process the image
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((128, 128))  # Resize to 224x224 for VGG19
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)  # Add batch dimension
    input_img = input_img / 255.0  # Normalize image
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01[0]

# Route for homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to the 'uploads' folder
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Get the prediction
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
