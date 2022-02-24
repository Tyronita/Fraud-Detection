// Script which changes the graph when axis value is changed     
const scatterPlot = document.querySelector('#scatter') // Get image to load it in

document.querySelectorAll('.axis-selector').forEach(selectorEl => {
    selectorEl.addEventListener('change', async e => {

        // Get axis variable values from DOM
        if (e.target.id == 'x-axis') {
            var x = e.target.value;
            var y = document.querySelector('#y-axis').value;
        } else {
            var x = document.querySelector('#x-axis').value;
            var y = e.target.value;
        }

        // Data to create the graph
        const queryString = '?x_axis=' + x + '&y_axis=' + y;
        const URLaddress = server + '/getScatter' + queryString;

        // Make POST Request to receive the image data for the new graph
        fetch( URLaddress)
            .then(response => {
                // GET Request successful - update image
                return response.json();  
            })
            .then( data => {
                scatterPlot.src = 'data:image/png;base64,' + data.imgB64;
                console.log("Image updated.");
            })
            .catch(error => {
                // GET Request unsuccessful
                console.log(error);
            });
        })
});

// Script which changes the graph when axis value is changed     
const timeseriesPlot = document.querySelector('#timeseries') // Get image to load it in

document.querySelector('#date-selector').addEventListener('change', async e => {

    // Get date values via DOM and add it to querystring.
    const queryString = '?date=' + e.target.value;
    const URLaddress = server + '/getTimeseries' + queryString;

    // Make POST Request to receive the image data for the new graph
    fetch(URLaddress)
        .then(response => {
            // GET Request successful - update image
            return response.json();  
        })
        .then( data => {
            timeseriesPlot.src = 'data:image/png;base64,' + data.imgB64;
            console.log("Image updated.");
        })
        .catch(error => {
            // GET Request unsuccessful
            console.log(error);
        });
});