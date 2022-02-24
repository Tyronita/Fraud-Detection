# Fraud Detection With Machine Learning

This program allows a user to manage a real database of credit card transactions of which some have been classified as fraud. <br>
On the server at launch, a logistic regression method is used to estimate the probability of fraudulency for each transaction; <br> the parameters for this method are tuned by the gradient descent algorithm. Any probabilities less than 40% are non-fraudulent and any above 60% are.
It is up to the user (analyst) to classify the transactions in the uncertainty region on the decision boundary using the scatter plots and timeseries visualizations.
The actual results can be revealed by pressing the red reset button in the top right hand corner of the page. 

## Setup

1. Clone this repository and install both Python then pip.

2. Go to Web App\Dataset directory on file explorer, extract creditcard.zip, and move the file creditcard.csv into dataset folder. 
   This large file was compressed so the repository didn't exceed Github's maximum file size limit of 100mb.
   
3. Pip install the 'virtualenv' package.
 
4. Navigate to the repository download using "cd ...\Fraud Detection"

5. Activate the virtual environment using the command: "Web App\env\Scripts\activate".

6. Move into the main webapp directory using "cd Webapp".

7. Run the program with "python app.py".

8. Wait until the url for the webapp to appear, and copy it from the command line into the browser.
