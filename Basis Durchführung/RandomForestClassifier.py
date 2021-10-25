import pandas as pd 
df=pd.read_csv('gdrive/My Drive/data.csv', delimiter=';')
df = df.drop(['sid', 'ort', 'appuser', 'versch_1', 'versch_2', 'cov_test_self_yn', 'cov_test_self_c', 'cov_test_self_c_recode', 'cov_test_self2_c', 'cov_test_self2_txt', 'cov_test_self2_c_recode', 'cov_test_employer_c', 'cov_test_self_amount_c'], axis=1)

df.rename(columns={'sex': 'isFemale'}, inplace=True)

df = df.replace({"yes": 1, "no": 0})
df = df.replace({"Women": 1, "Men": 0})

df['age'] = df['age'].str.replace(',','.')
df['age'] = df['age'].astype(float)

df = pd.get_dummies(df, columns = ['employed', 'homeoffice'])
pd.set_option("display.max_rows", 10, "display.max_columns", None)
df.dropna(inplace=True)

y=df.covid_test.values 
X=df.drop('covid_test',axis=1)


#Bewertung der Features
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#Festlegen der Features
df = df[['covid_test', 'age','homeoffice_no homeoffice', 'homeoffice_exclusively', 'work_7', 'hyper', 'homeoffice_less than half', 'migrants', 'adipos', 'cancer', 'vaccep', 'dyslip', 'employed_parttime', 'cov_impf', 'homeoffice_more than half', 'versch', 'cov_bkreis', 'employed_fultime', 'inland', 'q_distance_bin', 'versch_1_bin']]

#Aufteilung in Target und Features
y=df.covid_test.values 
X=df.drop('covid_test',axis=1)

#Training des Modells ohne Parameter
X_scaled =  StandardScaler().fit_transform(X) 

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled,y,test_size = 0.2,stratify=y, random_state=1)

classifier = RandomForestClassifier()
 
classifier.fit(X_train,y_train)

# make prediction 
preds = classifier.predict(X_valid) 
# check performance
accuracy_score(preds,y_valid) 
precision_score(preds,y_valid)
recall_score(preds,y_valid)
f1_score(preds,y_valid)