import React, { Component } from 'react';
import Header from './Header';
import get from 'lodash/get';
import './App.css';

class MatchupPrediction extends Component {
  constructor(props) {
    super(props);
    this.state = {
      game_predictions: undefined
    };

    // this.handleChange = this.handleChange.bind(this);
  }


  /**
   * NAME
   *    handleOnClick
   * 
   * PARAMS
   *    event, event data required for button onClick function
   * 
   * DESCRIPTION
   *    Send request through API structure to get prediction for selected matchup.
   * 
   * AUTHOR
   *    - Brandan Quinn
   * 
   * DATE
   *    5/1/19 4:21pm
   */
  handleOnClick = event => {
    let t1 = document.getElementById("t1");
    let t2 = document.getElementById("t2");
    let t1_id = t1.options[t1.selectedIndex].value;
    let t2_id = t2.options[t2.selectedIndex].value;

    fetch('/matchup-predict', {
        method: 'POST',
        body: JSON.stringify({
            t1: t1_id,
            t2: t2_id
        })
    })
    .then(res => res.json())
    .then(game_predictions => this.setState( {game_predictions} ));
    // console.log(document.getElementById("t1"))
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
        <div class="box">
            <div class="teams" key={get(game_pred, 'playing')}>{get(game_pred, 'playing')}</div>
            <div class="outcome" key={get(game_pred, 'prediction-message')}>{get(game_pred, 'prediction_message')}</div>
        </div>
        ))}
    </>
    );

    /**
     * NAME
     *  TeamSelect
     * 
     * DESCRIPTION
     *  Reusable drop down menu featuring every NBA team and their matching team id for NBA API purposes.
     * 
     * AUTHOR
     *  - Brandan Quinn
     * 
     * DATE
     *  5/1/19 3:49pm
     * @param {String} id, ID tag to get selected values from HTML document 
     */
    const TeamSelect = ({id}) => {
        return (<select id={id}>
            <option value="1610612737">Atlanta Hawks</option>
            <option value="1610612738">Boston Celtics</option>
            <option value="1610612751">Brooklyn Nets</option>
            <option value="1610612766">Charlotte Hornets</option>
            <option value="1610612741">Chicago Bulls</option>
            <option value="1610612739">Cleveland Cavaliers</option>
            <option value="1610612742">Dallas Mavericks</option>
            <option value="1610612743">Denver Nuggets</option>
            <option value="1610612765">Detroit Pistons</option>
            <option value="1610612744">Golden State Warriors</option>
            <option value="1610612745">Houston Rockets</option>
            <option value="1610612754">Indiana Pacers</option>
            <option value="1610612746">LA Clippers</option>
            <option value="1610612747">LA Lakers</option>
            <option value="1610612763">Memphis Grizzlies</option>
            <option value="1610612748">Miami Heat</option>
            <option value="1610612749">Milwaukee Bucks</option>
            <option value="1610612750">Minnesota Timberwolves</option>
            <option value="1610612740">New Orleans Pelicans</option>
            <option value="1610612752">NY Knicks</option>
            <option value="1610612760">OKC Thunder</option>     
            <option value="1610612753">Orlando Magic</option>  
            <option value="1610612755">Philadelphia 76ers</option>
            <option value="1610612756">Phoenix Suns</option>
            <option value="1610612757">Portland Trail Blazers</option>
            <option value="1610612758">Sacramento Kings</option>
            <option value="1610612759">San Antonio Spurs</option>
            <option value="1610612761">Toronto Raptors</option>
            <option value="1610612762">Utah Jazz</option>
            <option value="1610612764">Washington Wizards</option>
        </select>)
    }
    return (
      <div>
        <TeamSelect id="t1"/>
        vs.      
        <TeamSelect id="t2"/>

        <button onClick={this.handleOnClick}>Get Prediction</button>

        <hr />

        {this.state.game_predictions ? <PredictBox game_preds={this.state.game_predictions}/> : "Select a matchup and submit."}
      </div>
    );
  }
}

export default MatchupPrediction;
