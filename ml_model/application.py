# Brandan Quinn
# Senior Project
# Flask API

from flask import Flask
from flask import request
from flask import jsonify

application = Flask(__name__)

import utils

from model import train_model
from model import get_predictions

persistent_model = train_model()


@application.route('/predict', methods=['GET', 'POST'])
def predict_games():
    """
        Defines functionality for '/predict' endpoint. 
        If GET request is received, gets season averages for each game played at todays date, processes the data through model,
        and returns list of predictions.
        If POST request is received, gets season averages for each game played at given date, processes the data through model, 
        and returns list of predictions.

        :return: Returns JSON response containing list of predictions.

        - Brandan Quinn
        2/4/19 3:42pm
    """
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
        predictions_to_return = utils.assess_accuracy(date_string, predictions_to_return)

    return jsonify(predictions=predictions_to_return)


@application.route('/matchup', methods=['POST'])
def predict_matchup():
    """
        Defines functionality for '/matchup' endpoint.
        If request is received, gets season averages for the teams sent in the body of the request.
        Processes this data through the model and returns prediction for matchup to web application.

        :return: Returns JSON response containing prediction.

        - Brandan Quinn
        5/1/19 5:30pm
    """
    predictions_to_return = {}

    t1 = request.get_json().get('t1')
    t2 = request.get_json().get('t2')
    utils.predict_matchup(t1, t2)
    predictions_to_return = get_predictions(persistent_model)

    return jsonify(predictions=predictions_to_return)
