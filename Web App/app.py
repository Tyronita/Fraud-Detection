# import libraries/framework
from flask import Flask, render_template, url_for, request, Response, jsonify
import pandas as pd
import os, json, numpy, time, datetime, io, random
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# import functions from other parts of our program
import model, graphs

app = Flask(__name__)

# Classify case - responding to input
@app.route('/getScatter', methods = ['GET'])
def getScatter():

    # Import the decisionsDB
    decisionDB = pd.read_csv("dataset/decisionDB.csv")
    
    # GET querystring arguments from request
    x_axis = request.args['x_axis']
    y_axis = request.args['y_axis']

    # Plot the graph and generate the encoded image data for it 
    scatterB64 = graphs.createScatter(decisionDB, x_axis, y_axis).decode('utf8')

    # Return image data
    return jsonify({'imgB64': scatterB64})

# Classify case - responding to input
@app.route('/getTimeseries', methods = ['GET'])
def getTimeseries():

    # Import the decisionsDB
    decisionDB = pd.read_csv("dataset/decisionDB.csv")
    
    # GET querystring arguments from request
    selectedDate = request.args['date']

    # Plot the graph and generate the encoded image data for it 
    timeseriesB64 = graphs.createTimeseries(decisionDB, selectedDate).decode('utf8')

    # Return image data
    return jsonify({'imgB64': timeseriesB64})

# Classify case - responding to input
@app.route('/classify', methods = ['POST'])
def classify():

    print(request.get_json())

    # Import the decisionsDB
    decisionDB = pd.read_csv("dataset/decisionDB.csv")
    
    # GET data from payload stored as JSON
    data = request.get_json()
        
    transactionID = int(float(data['transactionID']))
    classification = data['classification']

    # Update the classifcation in dataframe
    index = decisionDB.index[decisionDB['ID'] == transactionID]

    decisionDB.loc[index, "Decided_Class"] = classification 
    #decisionDB.at[index, "Decided_Class"] = classification  

    # Save to csv to prevent future wasted time
    decisionDB.to_csv("dataset/decisionDB.csv", index=False)

    # Return sucess message
    return f'Success. Transaction with ID {transactionID } has been classified as {classification}.'

@app.route('/')
def index():
    
    # Check if a model has already been trained and a set has already been tested
    if os.path.exists("dataset/decisionDB.csv"):

        # bring in pre-tested dataset
        decisionDB = pd.read_csv("dataset/decisionDB.csv")

    else: # doesnt exist so train/test the model
        
        transactions = pd.read_csv("dataset/creditcard.csv") # Import the large datset locally
        transactions["V0"] = 1 # assign the 0th feature
        transactions['ID'] = list(range(1, len(transactions) + 1)) # assign unique IDs/indexes to every record
        
        dataset = model.balanceDataset(transactions) # balance/reduce dataset
        features = ['V0', 'V14', 'V20', 'V26', 'V21']  # define some test model

        testedSet = model.trainTest(dataset, features)
        decisionDB = model.createDecisionDB(testedSet) # make the decision database
        decisionDB.to_csv("dataset/decisionDB.csv", index=False) # save as csv to prevent future wasted time

    # copy dataset for filtering purposes
    filteredDB = decisionDB.copy()

    # test to see if querystring loaded in
    arguments = request.args.to_dict()
    if arguments:

        # filter out unwanted clasifcations
        if 'fraud' not in arguments:
            filteredDB = filteredDB[filteredDB["Decided_Class"] != 1]

        if 'non_fraud' not in arguments:
            filteredDB = filteredDB[filteredDB["Decided_Class"] != 0]

        if 'undecided' not in arguments:
            filteredDB = filteredDB[filteredDB["Decided_Class"] != numpy.nan]

        # filter by datetime

        # check if any datetimes filters enetered
        if arguments["start_date"] != '' or arguments["end_date"] != '' or arguments["start_time"] != '' or arguments["end_time"]:

            # variably set the start and end of the dataset
            dataset_start, dataset_end = "2013-09-01", "2013-09-02"
            
            # if min or max amount isn't present - set it to the start or end point of range of possible values
            if arguments["start_date"] == '':
                arguments["start_date"] = dataset_start # start of dataset

            if arguments["end_date"] == '':
                arguments["end_date"] = dataset_end # start of dataset
            
            if arguments["start_time"] == '':
                arguments["start_time"] = "00:00:00" # start of day

            if arguments["end_time"] == '':
                arguments["end_time"] = "23:59:59" # end of day

            # create a new temporary datetime column for filtering - needed in both searches
            filteredDB["Datetime"] = pd.to_datetime(filteredDB['Date'] + ' ' + filteredDB['Time'])

            # use the correct search type.
            if arguments["time_search_type"] == "entire_period":

                # create the start and end datetimes of the period we want to gett
                startDtStr = arguments["start_date"] + ' ' + arguments["start_time"]
                start_datetime = pd.to_datetime(startDtStr)

                endDtStr = arguments["end_date"] + ' ' + arguments["end_time"]
                end_datetime = pd.to_datetime(endDtStr)

                # create mask to filter between two datetimes
                mask = (start_datetime <= filteredDB["Datetime"]) & (filteredDB['Datetime'] <= end_datetime)

                # filter by datetime
                filteredDB = filteredDB[mask]

            else: # daily interval search

                # first filter out the dates we're not interested in
                start_date, end_date = arguments["start_date"], arguments["end_date"]

                #greater than the start date and smaller than the end date
                mask = (start_date <= filteredDB["Date"] ) & (filteredDB["Date"] <= end_date)

                # get rid of unwanted days
                filteredDB = filteredDB[mask] 

                start_time, end_time = arguments["start_time"], arguments["end_time"]

                # create an index of times Pandas can look at.
                timeIndex = pd.DatetimeIndex(filteredDB['Datetime'])

                # filter by the daily interval of time
                filteredDB = filteredDB.iloc[timeIndex.indexer_between_time(start_time, end_time)]

        # check if amount filter enetered
        if arguments["min_amount"] != '' or arguments["max_amount"] != '':

            # if min or max amount isn't present:
            # set it to the start or end point of range of possible values

            if arguments["min_amount"] == '':
                arguments["min_amount"] = 0

            elif arguments["max_amount"] == '':
                arguments["max_amount"] = "inf"
            
            min_amount = float(arguments["min_amount"])
            max_amount = float(arguments["max_amount"])

            # filter by amount
            filteredDB = filteredDB[filteredDB["Amount"].between(min_amount, max_amount, inclusive = True)]

        # check if probability filter enetered
        if arguments["min_probability"] != '' or arguments["max_probability"] != '':

            if arguments["min_probability"] == '':
                arguments["min_probability"] = 0

            elif arguments["max_probability"] == '':
                arguments["max_probability"] = 1.0
            
            min_probability = float(arguments["min_probability"])
            max_probability = float(arguments["max_probability"])

            # filter by probability
            filteredDB = filteredDB[filteredDB["Predicted_Prob"].between(min_probability, max_probability, inclusive = True)]

    # splitting table into column values we will show in the modal window and those that will be shown in the table
    profilesColNames = ['ID', 'Date', 'Time', 'Amount', 'Predicted_Prob', 'Decided_Class']
    detailsColNames = ['ID'] + [f'V{num}' for num in range(1, 29)] # features V1 - 28

    # select these columns from the filtered section of the databse
    profilesTable, detailsTable = filteredDB[profilesColNames], filteredDB[detailsColNames]

    # create the image data for the plots of the database
    timeseriesB64 = graphs.createTimeseries(decisionDB, '2013-09-02')
    scatterB64 = graphs.createScatter( decisionDB,'V14', 'V11')

    # render table and dashboard page
    return render_template('index.html',  row_data = list(profilesTable.values.tolist()), table_headings = profilesColNames, 
                            modal_data_table = list(detailsTable.values.tolist()), modal_data_headings = detailsColNames,
                            timeseriesImg = timeseriesB64.decode('utf8'), scatterImg = scatterB64.decode('utf8'), 
                            databaseLength = decisionDB.shape[0]) 

@app.route('/reset')
def reset():

    # bring in pre-tested dataset
    decisionDB = pd.read_csv("dataset/decisionDB.csv")

    # evaluate both predicted and decided
    predictedResults = model.evaluate(decisionDB, "Predicted")
    decidedResults = model.evaluate(decisionDB, "Decided")

    # create the confusion matrices for both predicted and decided
    predConfMatrixB64 = graphs.createConfusionMatrix(decisionDB, predictedResults ,'Predicted')
    decConfMatrixB64 = graphs.createConfusionMatrix(decisionDB, decidedResults, 'Decided')

    # build evaluation for both predicted and decided
    predictedEval = {
        "recall": model.recall(predictedResults),
        "precision": model.precision(predictedResults)
    }
    decidedEval = {
        "recall": model.recall(decidedResults),
        "precision": model.precision(decidedResults)
    }

    # reset current model
    transactions = pd.read_csv("dataset/creditcard.csv") # Import the large datset locally
    transactions["V0"] = 1 # assign the 0th feature
    transactions['ID'] = list(range(1, len(transactions) + 1)) # assign unique IDs/indexes to every record
    
    dataset = model.balanceDataset(transactions) # balance/reduce dataset
    features = ['V0', 'V14', 'V20', 'V26', 'V21'] # define some test model

    testedSet = model.trainTest(dataset, features)
    decisionDB = model.createDecisionDB(testedSet) # make the decision database
    decisionDB.to_csv("dataset/decisionDB.csv", index=False) # save as csv to prevent future wasted time

    # reset current model 
    return render_template('evaluation.html', predConfMatrixImg = predConfMatrixB64.decode('utf8'), modelEval = predictedEval,
                                            decConfMatrixImg = decConfMatrixB64.decode('utf8'), userEval = decidedEval)


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

# Update CSS every time - stops chaching
def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == "__main__":
    app.run(debug=True)
