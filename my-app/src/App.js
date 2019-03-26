import React, { Component } from 'react';
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
      .then(game_predictions => this.setState({ game_predictions }));
  }

  render() {
    return (
      <div className="App">
        <p>Hitting API</p>
        {JSON.stringify(this.state.game_predictions)}
      </div>
    );
  }
}

export default App;
