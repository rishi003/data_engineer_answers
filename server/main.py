# Flask application to serve model

from flask import Flask, request
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)
model: RandomForestRegressor = None


@app.route('/predict')
def predict():
    """
    A Flask route function that predicts a value based on two input parameters, vol_moving_avg and
    adj_close_rolling_med, obtained from request.args. The function then uses a pre-trained model
    to make a prediction for the given input parameters and returns the prediction as a string.
    The function does not handle any exceptions, and deepcode ignore XSS is added for development
    purposes only.

    Parameters:
    None

    Returns:
    A string representation of the prediction.

    Example:
    If the vol_moving_avg is 50 and adj_close_rolling_med is 100, the function will predict a value
    based on these inputs and return the prediction as a string.
    """
    global model
    if model is None:
        try:
            model = joblib.load("./data/model.joblib")
        except:
            return "Error loading model"
    vol_moving_avg = request.args.get('vol_moving_avg')
    adj_close_rolling_med = request.args.get('adj_close_rolling_med')
    prediction = model.predict(
        [[int(vol_moving_avg), int(adj_close_rolling_med)]])
    # deepcode ignore XSS: Only for Dev purpose
    return str(prediction[0])


# Load applicaiton and model
if __name__ == '__main__':
    # deepcode ignore RunWithDebugTrue: Only for Dev purpose
    app.run(host='0.0.0.0', port=5000, debug=True)
