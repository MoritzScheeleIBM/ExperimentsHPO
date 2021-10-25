from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#Load Dataset

X = df.drop(['covid_test'],axis=1)
 
y = df.covid_test.values 

X_scaled =  StandardScaler().fit_transform(X) 

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled,y,test_size = 0.2,stratify=y, random_state=1)


#Setting Hyperparameterspace
mlp_gs = MLPClassifier(max_iter=100)

random_grid = {'hidden_layer_sizes': [(10,30,10),(20,)],
               'activation': ['tanh', 'relu'],
               'solver': ['sgd', 'adam'],
               'alpha': [0.0001, 0.05],
               'learning_rate': ['constant','adaptive']}

mlp_gs_random = RandomizedSearchCV(estimator = mlp_gs, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
mlp_gs_random.fit(X_train, y_train)

#Model testing

y_pred = mlp_gs_random.predict(X_train)

print('Precision: %.3f' % precision_score(y_valid, y_pred, average='micro'))

print('Recall: %.3f' % recall_score(y_valid, y_pred, average='micro'))

print('Accuracy: %.3f' % accuracy_score(y_valid, y_pred))

print('F1 Score: %.3f' % f1_score(y_valid, y_pred, average='micro'))