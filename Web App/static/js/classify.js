const server = 'http://127.0.0.1:5000/'

// iterate through header cell values find index of the DecisionValues
headerCellEls = document.querySelectorAll('th')
for (i = 0; i < headerCellEls.length; i++) {
    if (headerCellEls[i].innerText.trim() == 'Decided_Class') {
        decidedClassColIdx = i
        break
    }
}

const classifyButtons = document.getElementsByClassName("classify-button");
for (i = 0; i < classifyButtons.length; i++) {
    classifyButtons[i].addEventListener("click", async function(e) {

        // Retrieve the data of the record to update from the HTML attrs
        let data = {
            'transactionID': e.target.dataset.transactionid, 
            'classification': e.target.dataset.classification
        };

        // Send a post request to update database 
        const response = await fetch(
            server + '/classify', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            }
        )
        // if request was succesful
        if (response.status == 200) {

            // Physically update row 
            const currCellEl = e.target.parentElement
            const currRowCellsEls = currCellEl.parentElement.children
            currRowCellsEls[decidedClassColIdx].innerText = e.target.dataset.classification

            // 

            console.log(response);
        
        // if reuqest was unsuccessful
        } else {
            alert("Couldn't update database");
        }               
    });
};