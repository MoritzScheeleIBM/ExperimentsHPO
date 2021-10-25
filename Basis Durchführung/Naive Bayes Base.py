import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

df=pd.read_csv('gdrive/My Drive/data.csv', delimiter=';')

#Aufteilung in Target und Features
y=df.covid_test.values 
X=df.drop('covid_test',axis=1)

#Training des Modells ohne Parameter
X_scaled =  StandardScaler().fit_transform(X) 

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled,y,test_size = 0.2,stratify=y, random_state=1)

classifier = MultinomialNB()
 
classifier.fit(X_train,y_train)

# make prediction 
preds = classifier.predict(X_valid) 
# check performance
accuracy_score(preds,y_valid) 
precision_score(preds,y_valid)
recall_score(preds,y_valid)
f1_score(preds,y_valid)