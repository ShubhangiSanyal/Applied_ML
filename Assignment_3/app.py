from flask import Flask, request, jsonify
from score import *
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
loaded_model = pickle.load(open("best_model.pkl", 'rb'))

@app.route('/score', methods=['POST'])
def main():
    # Get the text from the POST request
    #text = request.json.get('text')

    # Score the text using the loaded model
    prediction, propensity = score(request.json.get('text'), loaded_model, threshold=0.5)

    # Create the response JSON
    #response = {'prediction': prediction, 'propensity': propensity}

    # Return the response as JSON
    return jsonify({'prediction': prediction, 'propensity': propensity})


if __name__ == '__main__':
    app.run(debug=True)
