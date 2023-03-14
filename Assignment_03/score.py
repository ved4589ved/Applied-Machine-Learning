import numpy as np
# import pandas as pdde
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# import jobli
import pandas as pd
raw_data=pd.read_csv("RAW.csv")
train= pd.read_csv("train_ass3.csv")
test= pd.read_csv("test_ass3.csv")
validate= pd.read_csv("validate_ass3.csv")
X_train= train.Text
Y_train= train.Label
X_validate = validate.Text
Y_validate= validate.Label
X_test = test.Text
Y_test = test.Label

count = CountVectorizer().fit(raw_data.Text)
X_train=count.transform(X_train)
X_val = count.transform(X_validate)
X_test = count.transform(X_test)

tfidf_transform = TfidfTransformer()
tfidf_train = tfidf_transform.fit_transform(X_train)
tfidf_val= tfidf_transform.fit_transform(X_val)
tfidf_test = tfidf_transform.fit_transform(X_test)

Y_train= Y_train.astype('int')
Y_validate= Y_validate.astype('int')
Y_test = Y_test.astype('int')


# from sklearn.externals import joblib
import pickle

def text_vec(text):
    obs= count.transform([text])
    obs = tfidf_transform.fit_transform(obs)
    return obs

filename = open("test",'rb')
mlp =pickle.load(filename)

def score(text:str, model, threshold:float=0.5) -> (bool,float):
    # Transform the input text using the same used during training
    emb = text_vec(text)
    print(emb.shape)
    # Predict the propensity score for the input text for each class
    prediction=model.predict(emb)
    propensity = model.predict_proba(emb)
    return prediction[0], propensity[0]

print(score("You have won a free trip to Paris, click on link to redeem",mlp,0.4))