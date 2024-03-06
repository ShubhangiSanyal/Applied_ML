from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
import sklearn
from sklearn import svm
import pickle

train_df = pd.read_csv("train.csv")

#splitting the datframe into X and y
X_train = train_df['text']
y_train = train_df['spam']

tfidf = TfidfVectorizer(stop_words='english')

train_tfidf = tfidf.fit_transform(X_train)

loaded_model = pickle.load(open('best_model.pkl', 'rb'))

def score(text:str, model: sklearn.svm._classes.SVC, threshold:float) -> tuple:
    propensity = model.predict_proba(tfidf.transform([text]))[0]
    desired_predict = (model.predict_proba(tfidf.transform([text]))[:,1] >= threshold).astype(bool)
    #return (desired_predict[0], propensity)
    return (bool(desired_predict[0]), float(max(propensity)))