import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

news_data = pd.read_csv(r'D:\Projects\FakeNewsDetector\dataset\news_preprocessed.csv')

news_data['label'] = news_data['label'].map({'REAL': 0, 'FAKE': 1})

X=news_data['content'].values
Y=news_data['label'].values
vectorizer=TfidfVectorizer()  
vectorizer.fit(X)
X=vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y,random_state=2)

model=LogisticRegression()
model.fit(X_train,Y_train)

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


print('Accuracy on training data : ',training_data_accuracy*100)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy on test data : ',test_data_accuracy*100)

X_new = X_test[0]
prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is fake')
  
print(Y_test[0])

with open('fake_news_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)