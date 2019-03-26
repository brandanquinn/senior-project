import React, { Component } from 'react';
import get from 'lodash/get';
import logo from './logo.svg';
import './App.css';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      game_predictions: undefined
    };

    // this.handleChange = this.handleChange.bind(this);
  }
  
  componentDidMount() {
    fetch('/todays-games')
      // .then(res => res.json())
      .then(res => res.json())
      .then(game_predictions => this.setState( {game_predictions} ));
  }

  render() {
    const PredictBox = ({game_preds}) => (
      <>
        {game_preds.map(game_pred => (
          <div>
            <div className="team" key={get(game_pred, 'teams')}>{get(game_pred, 'teams')}</div>
            <div className="outcome" key={get(game_pred, 'predicted-outcome')}>{get(game_pred, 'predicted-outcome')}</div>
            <div className="pointdiff" key={get(game_pred, 'predicted-pointdiff')}>{get(game_pred, 'predicted-pointdiff')}</div>
          </div>
        ))}
      </>
    );

    return (
      <div className="App">
        <p>{this.state.game_predictions ? <PredictBox game_preds={get(this.state.game_predictions, "predictions")}/> : "fuck"}</p>
        {/* <p>{typeof get(this.state.game_predictions, 'predictions')}</p> */}
        {/* <p>{JSON.stringify(get(this.state.game_predictions, 'predictions'))}</p> */}
      </div>
      
    );
  }
}

export default App;
