import pandas as pd
import numpy as np

bl=pd.read_csv('/Users/moritzscheele/Desktop/GCS2021_EXT002_dummydata_20211020/data_GCS2021_EXT002_bl_dummy.csv', delimiter=';')
fu=pd.read_csv('/Users/moritzscheele/Desktop/GCS2021_EXT002_dummydata_20211020/data_GCS2021_EXT002_fu_dummy.csv', delimiter=';')
app=pd.read_csv('/Users/moritzscheele/Desktop/GCS2021_EXT002_dummydata_20211020/data_GCS2021_EXT002_app_dummy.csv', delimiter=';')

n = len(fu.index) - 1
i = 0
zeilebs = 0
while i <= n:
    if fu.iloc[i]['covid_test'] == "yes":
        sid = fu.iloc[i]['sid']
        zeilebs = bl[bl['sid'] == sid].index
        print(i, sid, zeilebs)
        bl.at['sid', zeilebs] = "yes"
    else:
        pass
    i = i + 1
print(bl['covid_test'])

i = 0
f = len(app.index) - 1
while i <= f:
    if app.iloc[i]['test_app'] == "yes":
        sid = app.iloc[i]['sid']
        zeilebs = bl[bl['sid'] == sid].index
        print(i, sid, zeilebs)
        bl.at['sid', zeilebs] = "yes"
    else:
        pass
    i = i + 1


print(bl['covid_test'])