# Brandan Quinn
# Senior Project
# Logistic Regression Model

import tensorflow as tf
from tensorflow import keras

import numpy as np

import json
import csv

import utils

import pandas as pd

column_names = [
    'LOCFLOAT',
    # 'POINTDIFF',
    'FG%',
    '3PT%',
    'OREB',
    'ASSISTS',
    'STEALS',
    'TURNOVERS'
]

def convert_location_to_float(a_loc):
    """
        Converts location of game (home/away) to float to be processed by Neural Network
        :param a_loc: String value (Home/Away) to be converted.
        :return: Returns floating point value to be processed by Neural Network.

        -Brandan Quinn
        10:58am 1/29/19
    """

    if a_loc == "Home":
        return 1.0
    else:
        return 0.0

def convert_location_to_str(a_loc):
    """
        Converts floating point values representing location back to string values for display

        :param a_loc: List of floating point locations processed by Neural Network
        :return: Returns newly created list of stringified locations.

        - Brandan Quinn
        11:04am 1/29/19
    """

    str_loc = []
    for loc in a_loc:
        if loc == 1.0:
            str_loc.append("Home")
        else:
            str_loc.append("Away")

    return str_loc

def build_model(a_train_data):
    """
        Initializes a Sequential Neural Network using the Tensorflow Keras API.

        :param a_train_data: Numpy array containing data sliced in order to train the model.
        :return: Trained Neural Network as an object to interact with and use for future predictions.

        - Brandan Quinn
        11:10am 1/29/19
    """

    model = keras.Sequential([
        keras.layers.Dense(
            64,
            activation=tf.nn.relu6,
            input_shape=(a_train_data.shape[1],)
        ),
        keras.layers.Dense(64, activation=tf.nn.relu6),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae'])
    return model

def measure_accuracy(a_scaled_predictions, a_labels):
    """
        Compares predictions to actual test data values to determine the accuracy of the model.
        If the result is within 3 points of the actual difference, I consider it to be a true prediction.

        :param a_scaled_predictions: List of point diff values predicted by the model.
        :param a_labels: List of actual point diff values from the dataset.
        :return: Returns nothing.

        - Brandan Quinn
        10:21am 1/30/19
    """

    score = 0
    for i in range(len(a_labels)):
        if a_scaled_predictions[i] > 0 and a_labels[i] > 0 or a_scaled_predictions[i] < 0 and a_labels[i] < 0 :
            score += 1
    
    print("Accuracy of general outcome: ", (score / len(a_labels)) * 100, "%")


def plot_history(history):
    """
        Plots a graph to display model's improvement while training.

        :param history: Variable representing the state of the model and how well it learned throughout training.
        :return: Returns nothing.

        - Brandan Quinn
        5:37pm 1/31/19
    """

    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error []')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
                label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
                label = 'Val loss')
    plt.legend()
    plt.ylim([0,0.5])
    plt.show()

def load_dataset(file_name):
    """
        Reads in csv dataset from local directory and slices
        necessary statistics in order for the ML model to be trained and tested.

        :param file_name: Name of file to read data from. 
        Can be passed either the original dataset file or the temporary prediction file.
        :return: Returns dict containing the data read in from the file.

        - Brandan Quinn
        5:50pm 1/27/19
    """

    teams = []
    opponents = []
    stats = []
    labels = []

    with open(file_name) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in read_csv:
            if line_count != 0:
                teams.append(row[1])
                opponents.append(row[5])
                stats.append([
                    convert_location_to_float(row[4]),
                    # (float(row[7])-float(row[8])),
                    (float(row[11])-float(row[27])),
                    (float(row[14])-float(row[30])),
                    (float(row[18])-float(row[34])),
                    (float(row[20])-float(row[36])),
                    (float(row[21])-float(row[37])),
                    (float(row[23])-float(row[39]))
                ])
                labels.append(float(row[7])-float(row[8]))
            line_count += 1

    csv_file.close()

    # Empty prediction file after running through
    f = open('nba.live.predict.csv', 'w+')
    f.write('\n')
    f.close()

    return {
        'teams': teams,
        'opponents': opponents,
        'stats': stats,
        'labels': labels
    }

def get_predictions(model):
    """
        Reads data from prediction csv file and uses trained model to predict pointdiff values.

        :param model: Trained Neural Network object to be used for prediction.
        :return: Returns list of predictions depending on number of games in the prediction dataset. 
        If dataset is empty (games not scheduled at given date), returns null list.

        - Brandan Quinn
        11:15am 2/4/19
    """
    predict_dataset = load_dataset('nba.live.predict.csv')
    live_teams = predict_dataset['teams']
    live_opponents = predict_dataset['opponents']
    live_stats = predict_dataset['stats']
    live_labels = predict_dataset['labels']
    live_data = [live_teams, live_opponents, live_stats, live_labels]
    recent_games = np.array(live_stats)

    # If no games are played at given date, return empty list.
    if recent_games.size == 0:
        return []

    game_predict = model.predict(recent_games).flatten()

    game_df = pd.DataFrame(live_stats, columns=column_names)
    game_df['POINTDIFF'] = game_predict
        
    game_df.insert(0, 'TEAM', live_teams)
    game_df.insert(1, 'OPPONENT', live_opponents)

    print(game_df.head(10))

    game_pred_obj_list = []

    outcomes = []
    for val in game_predict:
        if val > 0:
            outcomes.append('W')
        else:
            outcomes.append('L')

    count = 0
    for prediction in game_predict:
        game_pred_obj_list.append({
            't1': live_teams[count],
            't2': live_opponents[count],
            'predicted-outcome': outcomes[count],
            'predicted-pointdiff': str(game_predict[count])
        })
        count += 1

    return game_pred_obj_list


def train_model():
    """
        Prepares the data to be processed by the Neural Network, trains NN, and visualizes data/performance in console.

        :return: Returns trained model to be interacted with via application.
        - Brandan Quinn
        5:38pm 1/28/19
    """

    # load in dataset and split into arrays
    dataset = load_dataset('nba.games.stats.csv')
    teams = dataset['teams']
    opponents = dataset['opponents']
    stats = dataset['stats']
    labels = dataset['labels']
    data = [teams, opponents, stats, labels]

    data_len = len(teams)
    slice_pt = int(.8*data_len)

    # print("Elements of training data: ", train_slice)
    # print("Elements of testing data: ", test_slice)

    # slice training and testing data
    train_team_names = teams[:slice_pt]
    train_opp_names = opponents[:slice_pt]
    train_data = np.array(stats[:slice_pt])
    train_labels = labels[:slice_pt]
    # char_train_labels = labels[:slice_pt]
    # train_labels = convert_labels_to_float(char_train_labels)

    test_team_names = teams[slice_pt:]
    test_opp_names = opponents[slice_pt:]
    test_data = np.array(stats[slice_pt:])
    test_labels = labels[slice_pt:]
    # char_test_labels = labels[slice_pt:]
    # test_labels = convert_labels_to_float(char_test_labels)

    # create pandas dataframe to print sample data

    df = pd.DataFrame(train_data, columns=column_names)

    df['LABEL'] = train_labels
    df.insert(0, 'TEAM', train_team_names)
    df.insert(1, 'OPPONENT', train_opp_names)
    # df.insert(2, 'LOC', convert_location_to_str(train_data[0]))

    print(df.head(15))

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    # Saving copy of original test_data to display in table at the end of run.
    original_test_data = test_data
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # Seuquential model with two densely connected hidden layers.
    # Output layer returns single, continuous value.
    # Need to play with model a bit

    # model = build_model()
    # model.summary()

    # Display training progress for each completed epoch.
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0:
                print('')
            print('.', end='')

    EPOCHS = 200

    # Store training stats
    # history = model.fit(train_data, train_labels, epochs=EPOCHS,
    #                     validation_split=0.2, verbose=0,
    #                     callbacks=[PrintDot()])

    model = build_model(train_data)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0,
                        callbacks=[early_stop, PrintDot()])

    # plot_history(history)    

    # Evaluate the model using the testing data and get the total loss as well
    # as Mean Absolute Error - which is a common regression metric.
    # In this case, it represents the average difference
    # between predicted float value to represent win/loss and
    # actual result.
    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

    print("\nTesting set Mean Abs Error: {:7.2f}".format(mae))

    test_predictions = model.predict(test_data).flatten()
    measure_accuracy(test_predictions, test_labels)

    # Need to make sure predictions remain on the scale from 0-1.

    df = pd.DataFrame(original_test_data, columns=column_names)

    # Need to change output to print predictions mapped back to strings (W/L)
    # based on rounded predictions.
    df['OUTCOME'] = test_labels
    df['FLOATPRED'] = test_predictions
    df.insert(0, 'TEAM', test_team_names)
    df.insert(1, 'OPPONENT', test_opp_names)

    df_elements = df.sample(50)

    print(df_elements)

    return model
