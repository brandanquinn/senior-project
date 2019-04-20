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
        <Header />
      </div>
    );
  }
}

export default App;
