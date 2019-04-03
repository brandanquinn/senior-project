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

  render() {
    return (
      <div>
        <p>Home Page</p>
      </div>
      
    );
  }
}

export default App;
