const express = require('express');
const request = require('request');

const app = express();
const port = process.env.PORT || 3001;

/**
 * '/todays-games
 * 
 * NAME
 *  '/todays-games'
 * 
 * DESCRIPTION
 *  Sets up '/todays-games' route for web app. On loading of page, sends a GET request to python ML model.
 *  The model then returns list of predictions at today's date as JSON via response.
 *  The JSON responses are simply printed to web page without formatting for now.
 * 
 * AUTHOR
 *  Brandan Quinn
 * 
 * DATE
 *  5:47pm 2/6/19
 */
app.get('/todays-games', (req, res) => {
    request
    .get('http://127.0.0.1:5000/predict')
    .on('data', (chunk) => {
        const predictions_json = chunk.toString()
        res.send(predictions_json);
    })
    
});
/**
 * '/predict-by-date'
 * 
 * NAME
 *  '/predict-by-date'
 * 
 * DESCRIPTION
 *  Sets up '/predict-by-date' route for web app to send a POST request with a date string to python ML model.
 *  The model then returns a list of predictions as JSON at given date as a response.
 *  These JSON responses are just printed to the web page without formatting for now.
 * 
 *  Date string is currently sent as a query parameter.
 * 
 * AUTHOR
 *  Brandan Quinn
 * 
 * DATE
 *  2/6/19 5:42pm 
 */
app.get('/predict-by-date', (req, res) => {
    let date = req.query.date;
    request
    .post('http://127.0.0.1:5000/predict', { json: {date}})
    .on('data', (chunk) => {
        const predictions_json = chunk.toString()
        res.send(predictions_json);
    })
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`));