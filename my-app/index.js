const express = require('express');
const request = require('request');
const bodyParser = require('body-parser');

const get = require('lodash/get');

const {convert_prediction} = require('./utils/convert-predictions');

const app = express();
const port = process.env.PORT || 3001;

/**
 * '/todays-games'
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
        // Need to pre-process json
        const converted_predictions = get(JSON.parse(predictions_json), 'predictions').map(pred => convert_prediction(pred));
        res.send(converted_predictions);
    })
    
});

// var jsonParser = bodyParser.json()

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
 *  If query param is not sent, defaults to todays date.
 * 
 * AUTHOR
 *  Brandan Quinn
 * 
 * DATE
 *  2/6/19 5:42pm 
 */
app.post('/predict-by-date', (req, res) => {
    // let date = req.headers.referer.slice(-8);
    let body = '';
    req.on('data', chunk => {
        body += chunk.toString(); // convert Buffer to string
    });
    req.on('end', () => {
        console.log(body);
        // res.end('ok');

        let date = get(JSON.parse(body), 'date');

        console.log('DATE IS: ', date);
        request
        .post('http://127.0.0.1:5000/predict', { json: { date }})
        .on('data', (chunk) => {
            console.log(chunk.toString());
            const predictions_json = chunk.toString();
            const converted_predictions = get(JSON.parse(predictions_json), 'predictions').map(pred => convert_prediction(pred));
            console.log(converted_predictions);
            res.send(converted_predictions);
        })
    })
    
    
    // If date not passed as a query param, get todays games.
    
})

/**
 * '/matchup-predict'
 * 
 * NAME
 *  '/matchup-predict'
 * 
 * DESCRIPTION
 *  Sets up '/matchup-predict' route for web app to send a POST request with two team IDs to python ML model.
 *  The model then returns a prediction of a matchup between those two teams as a response.
 *  These JSON responses are just printed to the web page without formatting for now.
 * 
 *  Date string is currently sent as a query parameter.
 *  If query param is not sent, defaults to todays date.
 * 
 * AUTHOR
 *  Brandan Quinn
 * 
 * DATE
 *  5/1/19 4:34pm 
 */
app.post('/matchup-predict', (req, res) => {
    let body = '';
    req.on('data', chunk => {
        body += chunk.toString();
    });
    req.on('end', () => {
        let t1 = get(JSON.parse(body), 't1');
        let t2 = get(JSON.parse(body), 't2');

        request
        .post('http://127.0.0.1:5000/matchup', { json: { t1, t2 }})
        .on('data', (chunk) => {
            const predictions_json = chunk.toString();
            const converted_predictions = get(JSON.parse(predictions_json), 'predictions').map(pred => convert_prediction(pred));
            console.log(converted_predictions);
            res.send(converted_predictions);
        })
    });
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`));