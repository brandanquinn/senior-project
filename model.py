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
# load_dataset()

# NAME
#   load_dataset - reads in csv dataset from local directory and slices necessary 
#   statistics in order for the ML model to be trained and tested.

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
    labels = [] # win/loss

    with open('nba.games.stats.csv') as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in read_csv:
            if line_count != 0:
                teams.append(row[1])
                opponents.append(row[5])
                stats.append([(float(row[7])-float(row[8])), 
                    (float(row[11])-float(row[27])),
                    (float(row[14])-float(row[30])),
                    (float(row[19])-float(row[35])),
                    (float(row[20])-float(row[36])),
                    (float(row[23])-float(row[39]))
                    ])
                labels.append(row[6])
            line_count+=1

    csv_file.close()

    return {
        'teams': teams,
        'opponents': opponents,
        'stats': stats,
        'labels': labels
    } 

dataset = load_dataset()
teams = dataset['teams']
opponents = dataset['opponents']
stats = dataset['stats']
labels = dataset['labels']

print(teams[0])
print("vs.")
print(opponents[0])
print(stats[0])
print(labels[0])