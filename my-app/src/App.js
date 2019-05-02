import React, { Component } from 'react';
import Header from './Header';
import get from 'lodash/get';
import './App.css';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      game_predictions: undefined
    };

    // this.handleChange = this.handleChange.bind(this);
  }

  render() {
    return (
      <div>
        <div class="headbox">
          <Header />
        </div>
        <div>
          <p>Welcome to QML NBA Predictions developed by Brandan Quinn!</p>
          <p>This web application was built to provide NBA game predictions based on point differential.</p>
          <p>These predictions are generated using a Neural Network provided by Tensorflow's Keras API: </p>
          <p>If you click on the link to Today's Predictions in the menubar you will be greeted with predictions for each game played today.</p>
          <p>If no games appear, it is likely that there are no NBA basketball games played on that date.</p>
          <p>Similarly, if you select Predictions By Date, you will be able to input a date. Once the date is selected, you will be redirected to view the predictions at that date.</p>
          <hr />
        </div>
      </div>
    );
  }
}

export default App;
