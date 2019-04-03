const get = require('lodash/get');


/**
 * NAME: convert_prediction()
 * 
 * PURPOSE: Converts raw prediction JSON from Flask API and converts it into user-friendly info for React app.
 * 
 * PARAMS: 
 *  - game_pred, JSON object for each prediction
 *      - contains keys: t1 (team 1), t2 (team 2), predicted-outcome (W/L), and predicted-pointdiff(# val)
 * 
 * SUMMARY:
 *  If T1 predicted to Win against T2:
 *      Prediction Message will be: T1 will win by (Rounded point differential)
 *  Else:
 *      Prediction Message will be: T2 will win by (Absolute value of rounded point differential) 
 *          (absolute value needed as negative value indicates number of points T1 will lose by)
 * 
 * RETURNS:
 *  JSON containing Teams playing and predicted result message of the game.
 * 
 * AUTHOR:
 *  - Brandan Quinn
 * 
 * DATE:
 *  12:03pm 4/3/19
 */
exports.convert_prediction = (game_pred) => {
    // Pull off T1 and T2 

    // If predicted-outcome is W, print T1 will win by (predicted-pointdiff)
    // Else: print T2 will win by ((-1)*predicted-pointdiff)

    let prediction_message;

    if (get(game_pred, 'predicted-outcome') == 'W') {
        prediction_message = get(game_pred, 't1') + ' will win by: ' + Math.round(get(game_pred, 'predicted-pointdiff')) + 'pts';
    } else {
        prediction_message = get(game_pred, 't2') + ' will win by: ' + (-1 * Math.round(get(game_pred, 'predicted-pointdiff'))) + 'pts';   
    }

    return {
        'playing': get(game_pred, 't1') + ' vs. ' + get(game_pred, 't2'),
        prediction_message
    }
}