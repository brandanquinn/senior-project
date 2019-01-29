# Brandan Quinn
# Senior Project
# Logistic Regression Model

# Step 1:
#   - Load in dataset and slice accordingly

import tensorflow as tf
from tensorflow import keras

import numpy as np

import json
import csv

###
# convert_labels(labels)

# NAME
#   convert_labels - converts each string label to a float to be processed

# SYNOPSIS
#   a_labels:
#   - list of labels determining if a team won ("W") or lost ("L") a given game

# DESCRIPTION
#   Iterates through each element in a_labels and creates a new list
#   containing a float value that matches the outcome of the game.
#   1.0 if the team won, 0.0 if the team lost.

# RETURNS
#   Returns the newly converted floating point list of labels

# AUTHOR
#   Brandan Quinn

# DATE
#   6:13pm 1/28/19


def convert_labels(a_labels):
    float_labels = []
    for label in a_labels:
        if label == "W":
            float_labels.append(1.0)
        else:
            float_labels.append(0.0)

    return float_labels

###
# load_dataset()

# NAME
#   load_dataset - reads in csv dataset from local directory and slices
#   necessary statistics in order for the ML model to be trained and tested.

# SYNOPSIS
#   No args

# DESCRIPTION
#   Attempts to open csv file from local directory and reads
#   each line to lists in order to process through model.

#   The statistics used currently for training/testing are differences in
#       - points
#       - field goal percentage
#       - 3 point shot percentage
#       - total rebounds
#       - assists
#       - turnovers

# RETURNS
#   Returns a json object containing the data read in from the file

# AUTHOR
#   Brandan Quinn

# DATE
#   5:50pm 1/27/19

#
###


def load_dataset():
    teams = []
    opponents = []
    stats = []
    labels = []

    with open('nba.games.stats.csv') as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in read_csv:
            if line_count != 0:
                teams.append(row[1])
                opponents.append(row[5])
                stats.append([
                    (float(row[7])-float(row[8])),
                    (float(row[11])-float(row[27])),
                    (float(row[14])-float(row[30])),
                    (float(row[19])-float(row[35])),
                    (float(row[20])-float(row[36])),
                    (float(row[23])-float(row[39]))
                ])
                labels.append(row[6])
            line_count += 1

    csv_file.close()

    return {
        'teams': teams,
        'opponents': opponents,
        'stats': stats,
        'labels': labels
    }

###
# train_model()

# NAME
#   train_model - Prepares the data to be processed by the ML model.

# SYNOPSIS
#   No args

# DESCRIPTION
#   - Takes the loaded dataset object and splits it into separate lists.
#   - We then find the point to slice the data into training and testing sets
#       to be processed by the model.
#   - Print a dataframe table using the pandas package.
#   - Data is then normalized.

# RETURNS
#   None

# AUTHOR
#   Brandan Quinn

# DATE
#   5:38PM 1/28/19

#
###


def train_model():
    # load in dataset and split into arrays
    dataset = load_dataset()
    teams = dataset['teams']
    opponents = dataset['opponents']
    stats = dataset['stats']
    labels = dataset['labels']
    data = [teams, opponents, stats, labels]

    data_len = len(teams)
    slice_pt = int(.7*data_len)

    # print("Elements of training data: ", train_slice)
    # print("Elements of testing data: ", test_slice)

    # slice training and testing data
    train_team_names = teams[:slice_pt]
    train_opp_names = opponents[:slice_pt]
    train_data = np.array(stats[:slice_pt])
    char_train_labels = labels[:slice_pt]
    train_labels = convert_labels(char_train_labels)

    test_team_names = teams[slice_pt:]
    test_opp_names = opponents[slice_pt:]
    test_data = np.array(stats[slice_pt:])
    char_test_labels = labels[slice_pt:]
    test_labels = convert_labels(char_test_labels)

    # create pandas dataframe to print sample data
    import pandas as pd

    column_names = ['POINTDIFF', 'FG%', '3PT%', 'REB', 'ASSISTS', 'TURNOVERS']
    df = pd.DataFrame(train_data, columns=column_names)

    df['OUTCOME'] = train_labels
    df.insert(0, 'TEAM', train_team_names)
    df.insert(1, 'OPPONENT', train_opp_names)

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

    def build_model():
        model = keras.Sequential([
            keras.layers.Dense(
                64,
                activation=tf.nn.relu,
                input_shape=(train_data.shape[1],)
            ),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['mae'])
        return model

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

    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

    # plot_history(history)

    model = build_model()

    # history = model.fit(train_data, train_labels, epochs=EPOCHS,
    #                     validation_split=0.2, verbose=0,
    #                     callbacks=[early_stop, PrintDot()])

    # Evaluate the model using the testing data and get the total loss as well
    # as Mean Absolute Error - which is a common regression metric.
    # In this case, it represents the average difference
    # in prediction price to actual price.
    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

    print("\nTesting set Mean Abs Error: {:7.2f}".format(mae))

    test_predictions = model.predict(test_data).flatten()
    # Need to make sure predictions remain on the scale from 0-1.

    df = pd.DataFrame(original_test_data, columns=column_names)

    # Need to change output to print predictions mapped back to strings (W/L)
    # based on rounded predictions.
    df['OUTCOME'] = char_test_labels
    df['PREDICTION'] = test_predictions
    df.insert(0, 'TEAM', test_team_names)
    df.insert(1, 'OPPONENT', test_opp_names)

    print(df.head(15))

train_model()
