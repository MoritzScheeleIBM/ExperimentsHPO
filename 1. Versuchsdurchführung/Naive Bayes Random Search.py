from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

#Load Dataset

X = df.drop(['covid_test'],axis=1)
 
y = df.covid_test.values 

X_scaled =  StandardScaler().fit_transform(X) 

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled,y,test_size = 0.2,stratify=y, random_state=1)

#Setting Hyperparameterspace

mnb = MultinomialNB()

alpha = np.linspace(0,10,101)
random_grid = {'alpha': alpha,
               'fit_prior': [True, False]}

mnb_random = RandomizedSearchCV(estimator = mnb, param_distributions = random_grid, n_iter = 100, verbose=2, n_jobs = -1)# Fit the random search model
mnb_random.fit(X_train, y_train)

#Model testing

y_pred = mnb_random.predict(X_train)

print('Precision: %.3f' % precision_score(y_valid, y_pred, average='micro'))

print('Recall: %.3f' % recall_score(y_valid, y_pred, average='micro'))

print('Accuracy: %.3f' % accuracy_score(y_valid, y_pred))

print('F1 Score: %.3f' % f1_score(y_valid, y_pred, average='micro'))