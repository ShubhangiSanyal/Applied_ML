from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import pytest
from sklearn import svm
import sklearn
import requests
import app
from app import *
import os
from score import *

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

#splitting the datframe into X and y
X_train = train_df['text']
y_train = train_df['spam']
X_val = val_df['text']
y_val = val_df['spam']
X_test = test_df['text']
y_test = test_df['spam']

tfidf = TfidfVectorizer(stop_words='english')

train_tfidf = tfidf.fit_transform(X_train)

loaded_model = pickle.load(open('best_model.pkl', 'rb'))

from score import *

def test_score():
    threshold =0.5
    
    # Smoke Test - does the function produce some output without crashing
    prediction, propensity = score(X_test[10],loaded_model,threshold)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)
    
    # Format test - are the input/output formats/types as expected
    assert type(X_test[10]) == str
    assert type(loaded_model) == sklearn.svm._classes.SVC
    assert type(threshold) == float
    assert type(prediction) == bool
    assert type(propensity) == float
    
    # is prediction value 0 or 1
    assert (prediction in [0,1]) == True
    
    # is propensity score between 0 and 1
    assert (0<=propensity<=1) == True

    #When threshold is set to 0
    threshold = 0
    prediction, propensity = score(X_test[1],loaded_model,threshold) #testing on non spam
    assert prediction == True #to check if labels non spam also as spam
    prediction, propensity = score(X_test[10],loaded_model,threshold) #testing on spam
    assert prediction == True #to check if correct labels spam

    #When threshold is set to 1
    threshold = 1
    prediction, propensity = score(X_test[1],loaded_model,threshold)
    assert prediction == False  #to check if correct labels spam
    prediction, propensity = score(X_test[10],loaded_model,threshold)
    assert prediction == False #to check if labels spam also as non spam

    #assertion on Spam SMS
    print("Obvious Spam text")
    spam_text = "Buy now and get 50% off!"
    prediction, propensity = score(X_test[1], loaded_model, threshold=0.5)
    assert prediction == True

    #Assertion on non spam SMS
    print("Obvious Non-Spam text")
    non_spam_text = "Hello, how are you today?"
    prediction, propensity = score(X_test[0], loaded_model, threshold=0.5)
    assert prediction == False
    


import os
import pytest
import requests
import time
import threading
import subprocess

# Define the path to your Flask app file
FLASK_APP_PATH = 'C:/Users/shubh/Downloads/AML_Assignment3/app.py'

# Define the URL for the endpoint
ENDPOINT_URL = 'http://127.0.0.1:5000/score'

def launch_flask_app():
    
    # Function to launch the Flask app using the command line
    
    # Launch Flask app using command line
    subprocess.Popen(['python', FLASK_APP_PATH])

def close_flask_app():
    
    #Function to close the Flask app using the command line
    
    # Close Flask app using command line
    os.system("taskkill /f /im python.exe")  # Assumes Python process is running the Flask app

def test_flask():

    #Integration test function to test Flask app
    
    # Launch Flask app in a separate thread
    flask_thread = threading.Thread(target=launch_flask_app)
    flask_thread.start()

    # Wait for Flask app to start
    time.sleep(2)  # Adjust the time delay as needed

    try:
        # Send a test POST request to the Flask app
        data = {'text': 'There is a meeting tomorrow. Please be there.'}
        response = requests.post(ENDPOINT_URL, json=data)

        # Check the response
        assert response.status_code == 200
        assert 'prediction' in response.json()
        assert 'propensity' in response.json()

    finally:
        # Close Flask app after the test
        close_flask_app()

# Run the test function
if __name__ == '__main__':
    test_flask()