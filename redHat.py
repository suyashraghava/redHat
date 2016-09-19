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

from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
#ftp = SelectPercentile(f_classif, percentile=80)
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)
ftp = SelectFromModel(lsvc)
pactTrainFeatured = ftp.fit_transform(pactTrain.values, actTr.outcome)
features = ftp.get_support(indices = False)
c =[]
for i in range(len(features)): c.append(i) if ( features[i] == False) else 0
columNames = list(pactTest.columns.values)
print pactTest.columns[c]
pactTest.drop(pactTest.columns[c],axis = 1,inplace= True)
print pactTest.shape, pactTrainFeatured.shape

#activityTrain =np.array(pactTrain.activity_id)
#activityTrain = activityTrain.reshape(len(pactTrainFeatured),1)
#print activityTrain.shape, '\n'
#pactTrainFeatured = np.append(pactTrainFeatured,activityTrain,axis = 1 )
#pactTest['activity_id'] = te.activity_id
#print pactTest.shape,pactTrainFeatured.shape
#
#peopleTrain =np.array(pactTrain.people_id)
#peopleTrain = peopleTrain.reshape(len(pactTrainFeatured),1)
#print peopleTrain.shape, '\n'
#pactTrainFeatured = np.append(pactTrainFeatured,peopleTrain,axis = 1 )
#pactTest['people_id'] = te.people_id

import xgboost as xgb
dtrain = xgb.DMatrix(pactTrainFeatured,label=actTr.outcome)
dtest = xgb.DMatrix(pactTest.values)
param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['subsample'] = 0.7
param['colsample_bytree']= 0.7
param['min_child_weight'] = 0
param['booster'] = "gblinear"
watchlist  = [(dtrain,'train')]
num_round = 300
early_stopping_rounds=10
bst = xgb.train(param, dtrain, num_round ,watchlist,early_stopping_rounds=early_stopping_rounds)
#   PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 38 )
#pactTrainFeatured   = pca.fit_transform(pactTrainFeatured)
#xTest = pca.transform(pactTest.values)
print pactTrainFeatured.shape
#
##CROSS VALIDATION
#from sklearn import cross_validation
#pactTrainFeaturedCross,pactTestFeatured, y_Train, y_Test = cross_validation.train_test_split(pactTrainFeatured,actTr.outcome,test_size = 0.9,random_state = 0)
##   train model
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=50)
#clf = svm.SVC(probability = False)
#clf.fit(pactTrainFeaturedCross, y_Train)
##prob = clf.predict_proba(pactTest.values)
#y_pred = clf.predict_proba(pactTestFeatured)
#from sklearn.metrics import roc_auc_score
#print roc_auc_score(y_Test,y_pred[:,1])
#
#sys.exit()

# TRAIN REAL MODEL
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)

clf.fit(pactTrainFeatured, actTr.outcome)
prob = clf.predict_proba(pactTest.values)
#prob = clf.predict_proba(xTest)
xg = bst.predict(dtest)
#make submission file
output = pd.DataFrame({ 'activity_id' : actTe.activity_id, 'outcome': (prob[:,1] + xg)/2 })
output.to_csv('sollution.csv', index = False)

