from flask import Flask, request, jsonify
from score import *
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# train_df = pd.read_csv("train.csv")
# val_df = pd.read_csv("validation.csv")
# test_df = pd.read_csv("test.csv")

#splitting the datframe into X and y
# X_train = train_df['text']
# y_train = train_df['spam']
# X_val = val_df['text']
# y_val = val_df['spam']
# X_test = test_df['text']
# y_test = test_df['spam']


app = Flask(__name__)

# Load the trained model and vectorizer
loaded_model = pickle.load(open('best_model.pkl', 'rb'))
tfidf = TfidfVectorizer(stop_words='english')

@app.route('/score', methods=['POST'])
def predict_score():
    # Get the text from the POST request
    text = request.json.get('text')

    # Score the text using the loaded model
    prediction, propensity = score(text, loaded_model, threshold=0.5)

    # Create the response JSON
    response = {
        'prediction': prediction,
        'propensity_score': propensity
    }

    # Return the response as JSON
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(port=5000, debug=True, host='127.0.0.1')
