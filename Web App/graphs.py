from io import BytesIO
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib, base64
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def createTimeseries(decisionDB, selectedDate):

    # Get all transactions from that day
    filteredDB = decisionDB[decisionDB.Date == selectedDate]

    # Setting the date as the index since the TimeGrouper works on Index, the date column is not dropped to be able to count
    filteredDB['Datetime'] = pd.to_datetime(filteredDB['Datetime'])
    filteredDB.set_index('Datetime', drop=False, inplace=True)

    # Split by classification
    fraud = filteredDB[filteredDB.Decided_Class == 1]["Decided_Class"]
    non_fraud = filteredDB[filteredDB.Decided_Class == 0]["Decided_Class"]
    
    # Group transactions by hour, count them and plot it.
    fraud.groupby(pd.Grouper(freq='1H')).count().plot(kind='line', color="#FF8C00") # orange
    non_fraud.groupby(pd.Grouper(freq='1H')).count().plot(kind='line', color="#70a9e1") # light blue
    
    # label the axes
    plt.xlabel('')
    plt.ylabel('Frequency (per hour)')
    
    # Create legend
    plt.legend(["Fraud", "Non-fraud"])

    # load the image data into temporary file (IO)
    figfile = BytesIO()
    plt.savefig(figfile, format='png')

    # clear figure and rewind to beginning of file
    plt.clf()
    figfile.seek(0)  

    # Base64 encoding is a type of conversion of bytes into ASCII characters
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

# Plotting the data - .plot(...) for line & .scatter(...) for scatter
def createScatter(dataset, feature1, feature2):
    
    fraud = dataset[dataset.Decided_Class == 1]
    non_fraud = dataset[dataset.Decided_Class == 0]
    
    # Input features to plot
    plt.scatter(fraud[feature1], fraud[feature2], color = "#FF8C00")
    plt.scatter(non_fraud[feature1], non_fraud[feature2], color = "#70a9e1") 
    
    # label the axis
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    
    # Create legend
    plt.legend(["Fraud", "Non-fraud"])
    
    # load the image data into temporary file (IO)
    figfile = BytesIO()
    plt.savefig(figfile, format='png')

    # clear figure and rewind to beginning of file
    plt.clf()
    figfile.seek(0)  

    # Base64 encoding is a type of conversion of bytes into ASCII characters
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def createConfusionMatrix(dataset, testResults, classCategory):

    # create confusion matrix
    tp, tn, fp, fn = testResults
    cf_matrix = np.array([[tn, fp], [fn, fp]]) # make 2-d array (matrix)

    # add labels
    group_names = ["True Negative", "False Positive", "False Negative", "True Positive"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    # add percentages
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]

    # create heatmap
    labels = np.asarray(labels).reshape(2,2)
    sn.heatmap(cf_matrix, annot = labels, fmt="", cmap='Blues')
    
    # load the image data into temporary file (IO)
    figfile = BytesIO()
    plt.savefig(figfile, format='png')

    # clear figure and rewind to beginning of file
    plt.clf()
    figfile.seek(0)   

    # Base64 encoding is a type of conversion of bytes into ASCII characters
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

