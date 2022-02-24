# importing required libraries
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json, numpy, time, math, statistics, random, os

def balanceDataset(largeDataset):

    # Separating the portion of records that were fraulent
    fraud = largeDataset.loc[largeDataset.Class == 1]

    # Get the amount of fraudulent records
    fraudCount = len(fraud)

    # Take the random sample of non-fraudulent records
    non_fraud = largeDataset.loc[largeDataset.Class == 0].sample(fraudCount)

    frames = [fraud, non_fraud]
    dataset = pd.concat(frames).sample(frac=1).reset_index(drop=True) # shuffle again

    return dataset

def sigmoid(trainingExample, features: list, parameters: list) -> float:
    """
    j is the index of the feature out of our selected feature set.
    n is the number of selected features
    """
    z = 0 # our input to the sigmoid (probability) function
    n = len(features)
    for j in range(n):
        featureName = features[j]
        jthFeatureVal, jthParameterVal = trainingExample[featureName], parameters[j]
        z += jthFeatureVal * jthParameterVal  
    # large negative values cause a math range error
    if z < 0:
        return 1 - 1 / (1 + math.exp(z))
    else:
        return 1 / (1 + math.exp(-z))

def decision_boundary(probability):
    if probability > 0.5:
        return 1
    else:
        return 0

def cost(trainingSet, features: list, parameters: list) -> float:
    """
    m is the total number of training examples 
    """
    costSum = 0
    # iterate over dataframe by row
    for index, trainingExample in trainingSet.iterrows():
        probability = sigmoid(trainingExample, features, parameters)
        if trainingExample.Class == 1:
            # need to catch error if prediction so wrong it will be infinitely wrong -> predicted 0 but was 1
            try:
                costSum += -math.log(probability)
            except:
                corrected_probability = 10**-6 # nearly 0 
                costSum += -math.log(corrected_probability)
            # or vice versa if 1 but predicted 0
        else:
            try:
                costSum += -math.log(1 - probability)
            except:
                corrected_probability = 1 - 10**-6 # e.g nearly 1
                costSum += -math.log(1 - corrected_probability)
    # Makes an average
    m = trainingSet.shape[0]
    return costSum / m

def costPartialDerivative(trainingSet, jthFeature: str, features: list, parameters: list):
    """
    'jthFeature' is the name of the feature we're partially differentiating the cost function with respct to.
    'm' is the total number of training examples 
    """
    m = trainingSet.shape[0]
    pdSum = 0
    for index, trainingExample in trainingSet.iterrows():
        probability = sigmoid(trainingExample, features, parameters)
        jthFeatureVal = trainingExample[jthFeature]
        pdSum += (probability - trainingExample["Class"]) * jthFeatureVal
    return pdSum / m

def train(trainingSet, features: list):
    
    learningRate = 1
    
    # initialise all parameters at 0 as it's as good as being a random guess 
    parameters = [0 for x in features]
    
    # boolean flag indicating if the cost function has reached a minimum
    hasConverged = False
    
    # infinite loop but we break out when we reach convergence - this is subject to change
    while not hasConverged:
        
        # temp holders for calculated parameters
        newParameters = parameters
        
        # initialise values of the partial derviatives of the cost function (with the respect to the current parameter)
        # to null as they cannot be calculated yet
        pdValHistory = [None for x in features]
        
        # iteratively work out each parameter of index idx
        for j, jthParameter in enumerate(parameters):
            
            # Calculate the value of the partial derviatives of the cost function (with the respect to the current parameter)
            pdVal = costPartialDerivative(trainingSet, features[j], features, parameters)
            
            # Calculate the new value of the current parameter
            newParameter = jthParameter - (learningRate * pdVal)
            
            # update our new parameter and pd value
            newParameters[j] = newParameter
            pdValHistory[j] = pdVal
            
        # simultaneously update all our variables // we don't have to change pds as they will be recalulcated
        prevParameter = newParameters
                
        # checks for convergence
        if statistics.mean(pdValHistory) < 0.001:
            # breaks and exits program
            hasConverged = True
            
    return newParameters

# -------------------------------------------------------------------------------------------------#

def test (testingSet, features, tunedParameters):
    
    # Arrays of a predicted prob for each record in our testing set
    predictedProbList = []
    predictedClassList = []
    
    for idx, record in testingSet.iterrows():
        
        predictedProb = sigmoid(record, features, tunedParameters)
        
        # Use our tuned parameters and selected features to calculate a probability of a test example being fraudulent
        predictedProbList.append(predictedProb)
        
        # Use the probability to predict a class
        predictedClassList.append(decision_boundary(predictedProb))
    
    # Update this to our testing set dataframe
    testingSet['Predicted_Prob'] = predictedProbList
    testingSet['Predicted_Class'] = predictedClassList
    
    return testingSet      

def evaluate(testingSet, classType = "Predicted"):
    """
    Evaluate the amount of every correct and incroret instance of a predicted or decided classes.
    """
    
    # Count and class every correct and instance of a prediction
    tp = tn = fp = fn = 0 # where True Positive = tp, False Negative = fn, etc
    
    for index, testingExample in testingSet.iterrows():
        
        # Use the probability to predict a class
        predictedClass = testingSet.loc[index][f"{classType}_Class"]
        actualClass = testingSet.loc[index]["Class"]    
        
        if actualClass == predictedClass: # Got the prediction correction
            if actualClass == 1: # and it was a positive correct... etc.
                tp += 1 
            else:
                tn += 1
        else:
            if predictedClass == 1:
                fp += 1
            else:
                fn += 1
                
    return (tp, tn, fp, fn)

# % of all fraudulent transactions, predicted as actually fraudulent
def recall(testResults):
    tp, tn, fp, fn = testResults
    try:
        # true positives / true positives + false negatives
        return tp / (tp + fn)
    except:
        print("Can't calculate recall as there were no true positives or false negatives.")
        return None

# % of all transactions, which were actually fraudulent
def precision(testResults):
    tp, tn, fp, fn = testResults
    try:
        # true positives / true positives + false positive
        return tp / (tp + fp)
    except:
        print("Can't calculate precision as there were no true positives or false positives.")
        return None

# combines both metrics into one score
def f1_score(testResults):
    tp, tn, fp, fn = testResults
    if sum(testResults) != tn:
        return tp / (tp + 0.5 * (fp + fn))
    else:
        return None # zero error if total == tn: since -> tp = fp = fn = 0

#-----------------------------------------------------------------------#

def cross_validation(features, k, dataset):
    """
    Returns the average precision and recall values when training and testing a dataset k times.
    """
    
    # shuffle dataset to reduce bias
    dataset.sample(frac=1).reset_index(drop=True)
    
    sampleSize = len(dataset) // k
    folds = []

    # Split dataset into k folds
    for i in range(k):

        start = i * sampleSize 
        end = (i + 1) * sampleSize - 1 if i != k - 1 else len(dataset) - 1 # final index if last group
        folds.append(dataset.iloc[start: end]) # the- kth fold: testing)

    # Now our folds are created we can iterate through it:

    f1Scores = []

    for i in range(k):

        folds_copy = folds.copy() # create a copy of the folds

        # Create the training/test sets
        testingSet = folds_copy.pop(i) # kth fold
        trainingSet =  pd.concat(folds_copy) # remaining k-1 folds

        # Train our model
        tunedParameters = train(trainingSet, features)

        # Test our model
        testedSet = test(testingSet, features, tunedParameters)
        testResults = evaluate(testedSet)

        # Add results for future calculations 
        f1Scores.append(f1_score(testResults))

    return sum(f1Scores) / k

# -------------------------------------------------------------------------------------------------#

def selectFeatures(dataset):
    selectedFeatures = ["V0"]
    candidateFeatures = [f"V{num}"for num in range(1,29)] + ["Amount_Scaled", "Time_Scaled"]
    running = True
    
    currF1 = round(cross_validation(selectedFeatures, 2, dataset), 4)
    print("Current Model F1: ", currF1)
    
    while True: # runs until loop broken (when no better features found)
        modelImproved = False
        
        # test all features
        for possibleFeature in candidateFeatures:
            
            print("Testing:", possibleFeature)
            
            trialFeatures = selectedFeatures + [possibleFeature]
            newF1 = round(cross_validation(trialFeatures, 4, dataset), 4) # round to 4 d.p - essentially converged 
            
            # select the feature if score improved
            if newF1 > currF1:
                
                # remove from candidates
                selectedFeatures.append(possibleFeature)
                candidateFeatures.remove(possibleFeature)
                
                # update benchmark score
                currF1 = newF1
                print("Current Model: ", selectedFeatures, " gives F1 Score: ", currF1)
                
                # model disrupted -> loop again.
                modelImproved = True
                break
        
        if modelImproved == False:
            return selectedFeatures  # exits function  
# ------------------------------------------------------------------------------------

def trainTest (dataset, features):
    """
    Another pointless seeming HOWEVER for flask web app
    """
    
    # Training\testing set split
    midPoint = len(dataset) // 2
    
    # Create the sets
    trainingSet = dataset.iloc[:midPoint].copy()
    testingSet = dataset.iloc[midPoint:].copy()
    
    # Train the model and produce a set of tested data
    tunedParameters = train(trainingSet, features)
    testedSet = test(testingSet, features, tunedParameters)
    
    return testedSet

# -------------------------------------------------------------------

def createDecisionDB(testedSet):

    # copy dataset
    decisionDB = testedSet.copy()

    # Add a new column named Decided_Class by duplicating the predicted class column
    decisionDB['Decided_Class'] = decisionDB['Predicted_Class']

    for index, row in decisionDB.iterrows():
        
        # If a transaction has a predicted probability of less than 0.1 away from the decision boundary (0.50):
        if abs(decisionDB.loc[index,'Predicted_Prob'] - 0.50) < 0.1:
            
            decisionDB.at[index,'Decided_Class'] = None  # change it’s Decided_Class value to None.
    
    # Rename time column to time elapsed in preparation for actual time
    decisionDB = decisionDB.rename({"Time": "Elapsed_Time"}, axis='columns')
    timestamps = list(decisionDB['Elapsed_Time'])

    # Generate datetimes
    datetimes = generateDatetimes(timestamps)
    decisionDB['Datetime'] = datetimes
    
    # Add date and time to the data frame
    decisionDB['Date'] = [dt.date() for dt in datetimes]
    decisionDB['Time'] = [dt.time() for dt in datetimes]

    # Sort records with respect to time
    decisionDB = decisionDB.sort_values('ID') 

    return decisionDB

def generateDatetimes(timestamps):

    # get starting point in time of dataset
    startingDate = "01/09/2013"
    startTimestamp = time.mktime(datetime.strptime(startingDate, "%d/%m/%Y").timetuple())

    datetimes = []
    for timestamp in timestamps:

        # create datetime object from timestamp
        datetimeObj = datetime.fromtimestamp( startTimestamp + timestamp)
        datetimes.append(pd.to_datetime(datetimeObj))

    return datetimes