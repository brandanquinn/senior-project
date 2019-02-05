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

app.listen(port, () => console.log(`Example app listening on port ${port}!`));