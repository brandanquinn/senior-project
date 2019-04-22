# Brandan Quinn
# Senior Project
# Flask API

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

import utils

from model import train_model
from model import get_predictions

persistent_model = train_model()

"""
    Defines functionality for '/predict' endpoint. 
    If GET request is received, gets season averages for each game played at todays date, processes the data through model,
    and returns list of predictions.
    If POST request is received, gets season averages for each game played at given date, processes the data through model, 
    and returns list of predictions.

    :return: Returns JSON response containing list of predictions.
"""
@app.route('/predict', methods=['GET', 'POST'])
def predict_games():
    predictions_to_return = {}
    print('Receiving: ', request.method, ' request from API.')

    if request.method == 'GET':
        utils.predict(utils.get_todays_date())
        predictions_to_return = get_predictions(persistent_model)
    # TODO: Implement POST request to retrieve other days predictions. 
    elif request.method == 'POST':
        date_string = request.get_json().get('date')
        print('Date received: ', date_string)
        utils.predict(date_string)
        predictions_to_return = get_predictions(persistent_model)

    return jsonify(predictions=predictions_to_return)