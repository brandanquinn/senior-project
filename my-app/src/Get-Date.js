import React, { Component } from 'react';
import './App.css';


/**
 * NAME:
 *  GetDate
 *  - Derived object from React Component that is used to get a date from user to display predictions at said date.
 * 
 * SUMMARY:
 *  - Basic React Component that renders a form for users to interact with and input a date string.
 *  - On submission of date, loads the /predict-by-date 
 */
class GetDate extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div>
        <form action="/predict-by-date/">
            <p>Date string(YYYY/MM/DD):</p> <br/>
            <input type="date" name="date"/><br/>
            <input type="submit" value="Submit"/>
        </form>
      </div>
    );
  }
}

export default GetDate;
