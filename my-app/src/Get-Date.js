import React, { Component } from 'react';
import get from 'lodash/get';
import logo from './logo.svg';
import './App.css';

class GetDate extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div>
        <form action="/predict-by-date/">
            <p>Date string(YYYY/MM/DD):</p> <br/>
            <input type="text" name="date"/><br/>
            <input type="submit" value="Submit"/>
        </form>
      </div>
    );
  }
}

export default GetDate;
