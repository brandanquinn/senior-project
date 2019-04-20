exports.convert_date = (date) => {
    console.log('Date before conversion: ', date);
    // Need to remove dashes from date string
    return date.replace(/\-/g, '');
}