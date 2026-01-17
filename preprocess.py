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

nltk.download('stopwords')

news_data = pd.read_csv(r'D:\Projects\FakeNewsDetector\dataset\news.csv')

news_data = news_data.fillna('')

news_data['content'] =  ' ' + news_data['title'] + ' ' + news_data['text']
X = news_data['content'].values
y = news_data['label'].values

port_stem=PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in set(stopwords.words('english'))]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_data['content'] = news_data['content'].apply(stemming)
news_data.to_csv(r'D:\Projects\FakeNewsDetector\dataset\news_preprocessed.csv', index=False)