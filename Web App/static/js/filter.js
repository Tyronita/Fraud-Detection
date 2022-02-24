const errorEl = document.querySelector('#errorMessage')

document.getElementById('filterTableForm').addEventListener ("submit", e => {

    const inputs = e.target.elements
    let errors = []

    // throw an error if minimum value is more than maximum for our range functions

    // check both inputs entered
    if (inputs["min_amount"].value != "" && inputs["max_amount"].value != "") {

        // throw error if min more than max (amount)  
        if (inputs["min_amount"].value > inputs["max_amount"].value) {
            errors.push('The maximum amount value should not be smaller than the minimum value.')
        }

    }
    
    // check both inputs entered
    if (inputs["min_probability"].value != "" && inputs["max_probability"].value != "") {

        // throw error if min more than max (fraudulency probability)
        if (inputs["min_probability"].value > inputs["max_probability"].value) {
            errors.push('The maximum fraudulency probability value should not be smaller than the minimum value.')
        }

    }
     
    // first check: end date must be equal to or larger than start date
    if (Date.parse(inputs["start_date"].value) > Date.parse(inputs["end_date"].value)) {
        errors.push("The starting date should equal to or before the end date.")
    }
    // second check: if end date = start date then the end datetime must be after the start datetime
    else if (inputs["time_search_type"].value == "entire_period" && inputs["start_date"].value == inputs["end_date"].value
             && inputs["start_time"].value && inputs["end_time"].value) {

        // create the timestamps
        let date = inputs.start_date.value
        let start_timestamp = Date.parse(date + ' ' + inputs.start_time.value) 
        let end_timestamp = Date.parse(date + ' ' + inputs.end_time.value)
        
        // check if the start timestamp is greater than the end timestamp
        if (start_timestamp > end_timestamp) {
            errors.push("The end datetime must not be before the starting datetime for a period search.")
        }    
    }
    
    // Check for errors
    if (errors.length > 0) {
        errorEl.innerHTML = errors.join('<br>') // print errors to screen
        e.preventDefault() // stop form from submitting
        e.target.reset() // reset the inputs
    }

});

function atLeastOneChecked () {
    if (document.querySelectorAll('input[type=checkbox]:checked').length == 0) {
        errorEl.innerHTML = 'At least one classification of transaction must be selected to make a query.'
        return false
    } else {
        return true
    }
}
