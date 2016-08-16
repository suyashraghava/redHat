import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import preprocessing

def preprocess_people(data):
    data = data.drop(['date'], axis=1)
    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])
    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)
    #  Values in the people df is Booleans and Strings
    columns = list(data.columns)
    bools = columns[11:]
    strings = columns[1:11]
    for col in bools:
        data[col] = pd.to_numeric(data[col]).astype(int)
    for col in strings:
        data[col] = data[col].fillna('type 0')
        data[col] = data[col].apply(lambda x: x.split(' ')[1])
        data[col] = pd.to_numeric(data[col]).astype(int)
    return data


#read data
people = pd.read_csv('people.csv')
actTr = pd.read_csv('act_train.csv')
actTe = pd.read_csv('act_test.csv')

actTrain = actTr[['people_id','activity_id','date']]
actTest= actTe[['people_id','activity_id','date' ]]

#apply encoding
le = preprocessing.LabelEncoder()
actTrain['people_id'] = actTrain['people_id'].apply(lambda x: x.split('_')[1])
actTrain['people_id'] = pd.to_numeric(actTrain['people_id']).astype(int)
actTest['people_id'] = actTest['people_id'].apply(lambda x: x.split('_')[1])
actTest['people_id'] = pd.to_numeric(actTest['people_id']).astype(int)
te = actTest.iloc[:,1:].apply(le.fit_transform,axis=0)
tr = actTrain.iloc[:,1:].apply(le.fit_transform,axis=0)
te['people_id']= actTest['people_id']
tr['people_id']= actTrain['people_id']
people = preprocess_people(people)

#merge
pactTrain = pd.merge(people, tr , on='people_id')
pactTest= pd.merge(people, te, on='people_id')
print pactTest.shape, pactTrain.shape

#feature selection
from sklearn.feature_selection import SelectPercentile, f_classif
ftp = SelectPercentile(f_classif, percentile=10)
pactTrain = ftp.fit_transform(pactTrain.values, actTr.outcome)
features = ftp.get_support(indices = False)
c =[]
for i in range(len(features)): c.append(i) if ( features[i] == False) else 0
pactTest.drop(pactTest.columns[c],axis = 1,inplace= True)
print pactTest.shape, pactTrain.shape


#train model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)
clf.fit(pactTrain, actTr.outcome)
prob = clf.predict_proba(pactTest.values)

#make submission file
output = pd.DataFrame({ 'activity_id' : actTe.activity_id, 'outcome': prob[:,1] })
output.to_csv('sollution.csv', index = False)

