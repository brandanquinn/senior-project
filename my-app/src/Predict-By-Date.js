import React, { Component } from 'react';
import get from 'lodash/get';
import logo from './logo.svg';
import './App.css';
import { convert_date } from './utils/date-manip';

class PredictByDate extends Component {
    constructor(props) {
      super(props);
      this.state = {
        game_predictions: undefined
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
        {game_preds.map(game_pred => (
          <div class="box">
            <div class="teams" key={get(game_pred, 'playing')}>{get(game_pred, 'playing')}</div>
            <div class="outcome" key={get(game_pred, 'prediction-message')}>{get(game_pred, 'prediction_message')}</div>
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
