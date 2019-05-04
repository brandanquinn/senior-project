import React, { Component } from 'react';
import get from 'lodash/get';
import './App.css';
import { convert_date } from './utils/date-manip';

class PredictByDate extends Component {
    constructor(props) {
      super(props);
      this.state = {
        game_predictions: undefined,
        class_name: ""
        };
    }
    
    /**
     * NAME:
     *  get_params
     *  - Pulls the date query param from URL and constructs an object with it.
     * 
     * SUMMARY:
     *  - Constructs a URLSearchParams object using current window URL.
     *  - Get date key from urlParams object and return it as an object.
     * 
     * AUTHOR:
     *  Brandan Quinn
     * 
     * DATE:
     *  4/18/19 12:34pm
     */
    get_params = () => {
        var urlParams = new URLSearchParams(window.location.search);

        return {
            query: convert_date(urlParams.get('date')) || '',
        };
    }

    /**
     * NAME:
     *  change_class_name
     *  - If game predictions from date selected are for games that have already been played, change div classname to style PredictBoxes differently.
     * 
     * PARAMS:
     *  is_old_pred, Undefined / false if game has not been played yet.
     *  is_outcome_correct, Boolean value determining whether or not predicted outcome was correct.
     * 
     * SUMMARY:
     *  - If prediction is for game that has already been played, change class name var to reflect whether predicted outcome was correct.
     *  - Else: Reset class name var to empty string so that no additional css styles are added.
     * 
     * AUTHOR:
     *  - Brandan Quinn
     * 
     * DATE:
     *  5/4/19 2:36pm 
     */
    change_class_name = (is_old_pred, is_outcome_correct) => {
      if (is_old_pred) {
        is_outcome_correct ? this.class_name = "Correct" : this.class_name = "Incorrect";
      } else {
        this.class_name = "";
      }
    }
  
  componentDidMount() {
    fetch('/predict-by-date', {
        method: 'POST',
        body: JSON.stringify({
            date: get(this.get_params(), 'query')
        })
    })
    .then(res => res.json())
    .then(game_predictions => this.setState( {game_predictions} ));
  } 

  render() {
    /**
     * @param {object} game_preds
     *  - List of game predictons computed by ML model, received via API
     * 
     * NAME
     *  PredictBox
     *  - For each game prediction object, construct a object-oriented React component to display data neatly and clearly.
     * 
     * SUMMARY: 
     *  - Takes an object as a parameter and destructs object into list of objects (using { } notation)
     *  - For each prediction in game_preds, create a box in the overarching grid containing:
     *      - Team info
     *      - Outcome prediction
     *      - Point differential prediction
     * 
     * AUTHOR:
     *  Brandan Quinn
     * 
     * DATE:
     *  3/29/19 3:02pm
     */
    const PredictBox = ({game_preds}) => (
      <>
        {game_preds.length === 0 && <p>No games played today.</p>}
        {game_preds.map(game_pred => (
          <div>
            {this.change_class_name(get(game_pred, 'actual_result'), get(game_pred, 'is-outcome-correct'))}
            <div class={"box " + this.class_name}>
            <div class="teams" key={get(game_pred, 'playing')}>{get(game_pred, 'playing')}</div>
            <div class="outcome" key={get(game_pred, 'prediction-message')}>{get(game_pred, 'prediction_message')}</div>
            <hr/>
            {get(game_pred, 'actual_result') && 
            <div key={get(game_pred, "actual_result")}>{get(game_pred, 'actual_result')}</div>}
          </div>
          </div>
        ))}
      </>
    );

    return (
      <div>
        <p>Date of games: {get(this.get_params(), 'query')}</p>
        <div class="wrapper">
          {this.state.game_predictions ? <PredictBox game_preds={this.state.game_predictions}/> : "Loading"}
        </div>
        {/* <p>{typeof get(this.state.game_predictions, 'predictions')}</p> */}
        {/* <p>{JSON.stringify(get(this.state.game_predictions, 'predictions'))}</p> */}
      </div>
      
    );
  }
}

export default PredictByDate;
