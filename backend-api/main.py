from flask import Flask, request
from flask_cors import CORS
from apis.predict import predict_api

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def send_for_prediction():
    """
    Endpoint to send an image for prediction.

    Returns:
        str: JSON response containing the prediction results.
    """
    return predict_api(request.files)

if __name__ == '__main__':
    app.run(debug=True)
