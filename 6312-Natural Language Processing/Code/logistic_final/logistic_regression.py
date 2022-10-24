import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import os

path1 = os.path.split(os.getcwd())[0] + '/project_datasets'


np.random.seed(0)

df = pd.read_csv(path1+'/'+'new_df.csv')

print(df.isnull().sum())
df.dropna(inplace=True)

X = df['text'].values
y = df['target'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

tfidf= TfidfVectorizer()
tfidf.fit(X_train)
X_train_tfidf =tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf1 = LogisticRegression().fit(X_train_tfidf, y_train)

predicted1 = clf1.predict(X_test_tfidf)
from sklearn.metrics import accuracy_score

print('accuracy of the model:',accuracy_score(y_test, predicted1))
print('f1-score of the model:',metrics.f1_score(y_test, predicted1))
print(metrics.classification_report(y_test, predicted1))
print(metrics.confusion_matrix(y_test, predicted1))








