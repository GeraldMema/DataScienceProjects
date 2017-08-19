#libraries import
import csv

import pandas as pd
from sklearn import preprocessing, svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB

#datasets
dataTrainPath = 'C:\\Users\\user\\Documents\\Mine\\Projects\\DataMining\\ted2\\train.tsv'

#read train dataset
df = pd.read_csv(dataTrainPath, sep='\t')
#convert features to categorial values
at1 = pd.get_dummies(df['Attribute1'])
at3 = pd.get_dummies(df['Attribute3'])
at4 = pd.get_dummies(df['Attribute4'])
at6 = pd.get_dummies(df['Attribute6'])
at7 = pd.get_dummies(df['Attribute7'])
at8 = pd.get_dummies(df['Attribute8'])
at9 = pd.get_dummies(df['Attribute9'])
at10 = pd.get_dummies(df['Attribute10'])
at11 = pd.get_dummies(df['Attribute11'])
at12 = pd.get_dummies(df['Attribute12'])
at14 = pd.get_dummies(df['Attribute14'])
at15 = pd.get_dummies(df['Attribute15'])
at16 = pd.get_dummies(df['Attribute16'])
at17 = pd.get_dummies(df['Attribute17'])
at18 = pd.get_dummies(df['Attribute18'])
at19 = pd.get_dummies(df['Attribute19'])
at20 = pd.get_dummies(df['Attribute20'])
#convert numerical to categorial values
at2 = pd.cut(df['Attribute2'],bins=5)
at2 = pd.get_dummies(at2)
at5 = pd.cut(df['Attribute5'],bins=5)
at5 = pd.get_dummies(at5)
at13 = pd.cut(df['Attribute13'],bins=5)
at13 = pd.get_dummies(at13)

X_train = [at1,at2,at3,at4,at5,at6,at7,at8,at9,at10,at11,at12,at13,at14,at15,at16,at17,at18,at19,at20]
X = pd.concat(X_train,axis=1)
X = X.as_matrix()
Y = df['Label']
#classifiers
nbClf = MultinomialNB()
rfClf = RandomForestClassifier(n_estimators=100, warm_start=True)
svmClf = svm.SVC(kernel='rbf', C = 2.0, gamma='auto')

#cross validation
crossValidation = StratifiedKFold(Y, n_folds=10)
#evaluate accuracy
nbAc = 0
rfAc = 0
svmAc = 0
for i, (train, test) in enumerate(crossValidation):
    #NB
    nbClf.fit(X[train], Y[train])
    nbPred = nbClf.predict(X[test])
    ac = accuracy_score(Y[test], nbPred)
    nbAc = nbAc + ac
    #RF
    rfClf.fit(X[train], Y[train])
    rfPred = rfClf.predict(X[test])
    ac = accuracy_score(Y[test], rfPred)
    rfAc = rfAc + ac
    #SVM
    svmClf.fit(X[train], Y[train])
    svmPred = svmClf.predict(X[test])
    ac = accuracy_score(Y[test], svmPred)
    svmAc = svmAc + ac

naiveBayesAccuracy = float(nbAc) / 10
randomForestAccuracy = float(rfAc) / 10
supportVectorMachineAccuracy = float(svmAc) / 10

#write the results
result = [['Statistic Measure','Naive Bayes','Random Forest','SVM']]
accuracy = ['Accuracy']
accuracy.append(naiveBayesAccuracy)
accuracy.append(randomForestAccuracy)
accuracy.append(supportVectorMachineAccuracy)
result.append(accuracy)
with open('C:\\Users\\user\\PycharmProjects\\ted2\\output\\EvaluationMetric_10fold.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter='\t')
    a.writerows(result)