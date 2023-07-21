import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Importing Data
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')


# Data Preproccesing
nltk.download('stopwords')

corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review
              if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)

    corpus.append(review)

cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
Y = df.iloc[:, 1].values


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 24)


# Training Model
model = RandomForestClassifier(n_estimators= 2442, criterion= 'entropy')

model.fit(X_train, Y_train)

test_pred = model.predict(X_test)

print(f'Test Accuracy is : {accuracy_score(Y_test, test_pred)}')