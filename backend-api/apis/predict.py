from utils.model_creator import make_or_restore_model
from flask import request, jsonify
import os
import numpy as np
from utils.preprocess_image import preprocess_image

model = make_or_restore_model()
plant_class = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', \
               'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', \
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', \
               'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', \
               'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', \
               'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', \
               'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', \
               'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', \
               'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

def predict_api(uploaded_image):
    """
    Predict the type of disease and accuracy for an uploaded image.

    Args:
        uploaded_image (werkzeug.datastructures.FileStorage):
            The uploaded image file.

    Returns:
        jsonify: JSON response containing the accuracy and type of disease predicted by the model.
            If an error occurs during prediction, an error message is included in the response.
    """
    try:
        if 'image' not in uploaded_image:
            return jsonify({'error': 'No image provided'})

        image = uploaded_image['image']

        image.save('temp_image.jpg')

        processed_image = preprocess_image('temp_image.jpg')

        predictions = model.predict(processed_image)

        accuracy = str((np.max(predictions))*100)
        disease_type = str(plant_class[np.argmax(predictions)])

        os.remove('temp_image.jpg')

        return jsonify({'Accuracy': accuracy, 'Type of Disease': disease_type})
    except Exception as e:
        return jsonify({'error': str(e)})