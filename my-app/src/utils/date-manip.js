/**
 * NAME: convert_date
 *  - Convert date from automatically generated YYYY-MM-DD from picker to required config of YYYYMMDD, thus just removing '-'
 * 
 * SUMMARY:
 *  - Uses JS String.replace() function with global regex to replace all occurrences of '-' with empty char. (Thus removing them)
 * 
 * RETURNS:
 *  - Correct date string
 * 
 * AUTHOR:
 *  - Brandan Quinn
 * 
 * DATE:
 *  3:54pm 4/20/19
 */
exports.convert_date = (date) => {
    console.log('Date before conversion: ', date);
    // Need to remove dashes from date string
    return date.replace(/\-/g, '');
}