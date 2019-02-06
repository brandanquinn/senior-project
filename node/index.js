const express = require('express');
const request = require('request');

const app = express();
const port = process.env.PORT || 3001;

app.get('/', (req, res) => {
    request
    .get('http://127.0.0.1:5000/predict')
    .on('data', (chunk) => {
        const predictions_json = chunk.toString()
        res.send(predictions_json);
    })
    
});

app.get('/predict-by-date', (req, res) => {
    request
    .post('http://127.0.0.1:5000/predict', { json: {date: '20190210'}})
    .on('data', (chunk) => {
        const predictions_json = chunk.toString()
        res.send(predictions_json);
    })

    // request({
    //     url: 'http://127.0.0.1:5000/predict',
    //     method: 'POST',
    //     json: true,   // <--Very important!!!
    //     body: {date: '20190204'}
    // }, function (error, response, body){
    //     console.log(response);
    // });
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`));