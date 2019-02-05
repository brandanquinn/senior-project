# File used for testing purposes; can select to get team averages for game prediction or gather game data if game was already played.

import utils

user_input = input("(predict) or (gather) data? ")
date = ''

if user_input == "predict":
    date = utils.get_todays_date()
elif user_input == "gather":
    date = utils.get_yesterdays_date()
else:
    print("Invalid input. Try again.")
    exit()

game_list = utils.get_game_list(date)

for game in game_list:
    # print(game.keys())
    home_team = game.get('hTeam').get('triCode')
    away_team = game.get('vTeam').get('triCode')
    game_id = game.get('gameId')
    print('Checking game with id: ', game_id)
    utils.get_stats(game, date, game_id, home_team, away_team, user_input)