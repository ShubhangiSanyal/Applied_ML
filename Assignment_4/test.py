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
    
# INTEGRATION TEST

import os
import requests
import subprocess
import time
import unittest

class TestFlaskIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Launch Flask app using command line
        cls.flask_process = subprocess.Popen(['python', 'app.py']) 
        time.sleep(3)  # Give some time for the server to start

    def test_flask(self):
        # Test the response from the localhost endpoint
        data = {'text': 'There is a meeting tomorrow. Please be there.'}
        #stop_flask_app()
        response = requests.post('http://127.0.0.1:5000/score', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json())
        self.assertIn('propensity', response.json())
        
    @classmethod
    def tearDownClass(cls):
        # Close Flask app using command line
        cls.flask_process.terminate()


class TestDocker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the Docker image
        subprocess.run(['docker', 'build', '-t', 'my_flask_app', '.'], check=True)
        # Launch the Docker container
        cls.docker_process = subprocess.check_output(
            ["docker", "run", "-d", "-p", "5000:5000", "my_flask_app"]
        ).decode("utf-8").strip()
        time.sleep(5)  # Give some time for the server to start

    def test_docker(self):
        # Test the response from the localhost endpoint
        data = {'text': 'There is a meeting tomorrow. Please be there.'}
        response = requests.post('http://127.0.0.1:5000/score', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json())
        self.assertIn('propensity', response.json())
        
    @classmethod
    def tearDownClass(cls):
        # Close the Docker container
        subprocess.run(["docker", "stop", cls.docker_process], check=True)
        subprocess.run(["docker", "rm", cls.docker_process], check=True) 

if __name__ == '__main__':
    unittest.main()