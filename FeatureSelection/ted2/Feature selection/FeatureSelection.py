#libraries import
import csv

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

#datasets
dataTrainPath = 'C:\\Users\\user\\Documents\\Mine\\Projects\\DataMining\\ted2\\train.tsv'
dataTestPath = 'C:\\Users\\user\\Documents\\Mine\\Projects\\DataMining\\ted2\\test.tsv'

#read train dataset
df = pd.read_csv(dataTrainPath, sep='\t')
#read train dataset
dfTest = pd.read_csv(dataTestPath, sep='\t')
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

# Data preparation
X_train = [ at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12, at13, at14, at15, at16, at17, at18,
           at19,at20]
X = pd.concat(X_train, axis=1)
X = X.as_matrix()
Y = df['Label']

# first step - calculate the information gain for attribute i
model = ExtraTreesClassifier()
model.fit(X, Y)
allInfoGain = model.feature_importances_
informationGain = []
infGain1 = allInfoGain[0]+allInfoGain[1]+allInfoGain[2]+allInfoGain[3]
informationGain.append(infGain1)
infGain2 = allInfoGain[4]+allInfoGain[5]+allInfoGain[6]+allInfoGain[7]+allInfoGain[8]
informationGain.append(infGain2)
infGain3 = allInfoGain[9]+allInfoGain[10]+allInfoGain[11]+allInfoGain[12]+allInfoGain[13]
informationGain.append(infGain3)
infGain4 = allInfoGain[14]+allInfoGain[15]+allInfoGain[16]+allInfoGain[17]+allInfoGain[18]+allInfoGain[19]+allInfoGain[20]+allInfoGain[21]+allInfoGain[22]+allInfoGain[23]
informationGain.append(infGain4)
infGain5 = allInfoGain[24]+allInfoGain[25]+allInfoGain[26]+allInfoGain[27]+allInfoGain[28]
informationGain.append(infGain5)
infGain6 = allInfoGain[29]+allInfoGain[30]+allInfoGain[31]+allInfoGain[32]+allInfoGain[33]
informationGain.append(infGain6)
infGain7 = allInfoGain[34]+allInfoGain[35]+allInfoGain[36]+allInfoGain[37]+allInfoGain[38]
informationGain.append(infGain7)
infGain8 = allInfoGain[39]+allInfoGain[40]+allInfoGain[41]+allInfoGain[42]
informationGain.append(infGain8)
infGain9 = allInfoGain[43]+allInfoGain[44]+allInfoGain[45]+allInfoGain[46]
informationGain.append(infGain9)
infGain10 = allInfoGain[47]+allInfoGain[48]+allInfoGain[49]
informationGain.append(infGain10)
infGain11 = allInfoGain[50]+allInfoGain[51]+allInfoGain[52]+allInfoGain[53]
informationGain.append(infGain11)
infGain12 = allInfoGain[54]+allInfoGain[55]+allInfoGain[56]+allInfoGain[57]
informationGain.append(infGain12)
infGain13 = allInfoGain[58]+allInfoGain[59]+allInfoGain[60]+allInfoGain[61]+allInfoGain[62]
informationGain.append(infGain13)
infGain14 = allInfoGain[63]+allInfoGain[64]+allInfoGain[65]
informationGain.append(infGain14)
infGain15 = allInfoGain[66]+allInfoGain[67]+allInfoGain[68]
informationGain.append(infGain15)
infGain16 = allInfoGain[69]+allInfoGain[70]+allInfoGain[71]+allInfoGain[72]
informationGain.append(infGain16)
infGain17 = allInfoGain[73]+allInfoGain[74]+allInfoGain[75]+allInfoGain[76]
informationGain.append(infGain17)
infGain18 = allInfoGain[77]+allInfoGain[78]
informationGain.append(infGain18)
infGain19 = allInfoGain[79]+allInfoGain[80]
informationGain.append(infGain19)
infGain20 = allInfoGain[81]+allInfoGain[82]
informationGain.append(infGain20)

#here define the accuracy when we have all attributes -- without dimensionality reduction
accuracyForAllAtributes = 0.982716049382716
maxAccuracy = accuracyForAllAtributes
index = -1
for i in range(0,20):

    # RF classifier
    rfClf = RandomForestClassifier(n_estimators=100, warm_start=True)

    # cross validation
    crossValidation = StratifiedKFold(Y, n_folds=10)

    # Data preparation
    X_train = [at20, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12, at13, at14, at15, at16, at17, at18, at19]
    numberOfAttribute=i+1
    if i==19:
        i=-1
        numberOfAttribute=20
    X_train.pop(i+1)
    X = pd.concat(X_train, axis=1)
    X = X.as_matrix()

    #second step evaluate accuracy when we remove attribute i
    rfAc = 0
    for i, (train, test) in enumerate(crossValidation):
        #RF
        rfClf.fit(X[train], Y[train])
        rfPred = rfClf.predict(X[test])
        ac = accuracy_score(Y[test], rfPred)
        rfAc = rfAc + ac
    #RF accuracy after removing one feature
    randomForestAccuracy = float(rfAc) / 10
    #Make the plot
    informationGainComment = 'THE INFORMATION GAIN FOR THE REMOVING FEATURE (ATTRIBUTE'+ numberOfAttribute.__str__()+') IS : '+informationGain[numberOfAttribute-1].__str__()
    plt.title(informationGainComment)
    x=[1,2]
    y=[accuracyForAllAtributes,randomForestAccuracy]
    plt.xticks([1, 2])
    plt.axes().set_xticklabels(['ALL FEATURES ', 'REMOVE FEATURE '+numberOfAttribute.__str__()])
    if(accuracyForAllAtributes>randomForestAccuracy):
        plt.plot(x,y,'r')
    else:
        plt.plot(x, y, 'g')
    # plt.show()

    #final step is to run the algorithm with test.csv and we have to get the best accuracy from all steps
    if(randomForestAccuracy >= maxAccuracy):
        maxAccuracy = randomForestAccuracy
        index = numberOfAttribute - 1

#we run the algorithm with test set
print index
#train
X_train = [ at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12, at13, at14, at15, at16, at17, at18,
           at19,at20]
if(index != -1):
    X_train.pop(index)
X = pd.concat(X_train, axis=1)
X = X.as_matrix()
#test
at1 = pd.get_dummies(dfTest['Attribute1'])
at3 = pd.get_dummies(dfTest['Attribute3'])
at4 = pd.get_dummies(dfTest['Attribute4'])
at6 = pd.get_dummies(dfTest['Attribute6'])
at7 = pd.get_dummies(dfTest['Attribute7'])
at8 = pd.get_dummies(dfTest['Attribute8'])
at9 = pd.get_dummies(dfTest['Attribute9'])
at10 = pd.get_dummies(dfTest['Attribute10'])
at11 = pd.get_dummies(dfTest['Attribute11'])
at12 = pd.get_dummies(dfTest['Attribute12'])
at14 = pd.get_dummies(dfTest['Attribute14'])
at15 = pd.get_dummies(dfTest['Attribute15'])
at16 = pd.get_dummies(dfTest['Attribute16'])
at17 = pd.get_dummies(dfTest['Attribute17'])
at18 = pd.get_dummies(dfTest['Attribute18'])
at19 = pd.get_dummies(dfTest['Attribute19'])
at20 = pd.get_dummies(dfTest['Attribute20'])
#convert numerical to categorial values
at2 = pd.cut(dfTest['Attribute2'],bins=5)
at2 = pd.get_dummies(at2)
at5 = pd.cut(dfTest['Attribute5'],bins=5)
at5 = pd.get_dummies(at5)
at13 = pd.cut(dfTest['Attribute13'],bins=5)
at13 = pd.get_dummies(at13)
X_test = [ at1, at2, at3, at4, at5, at6, at7, at8, at9, at10, at11, at12, at13, at14, at15, at16, at17, at18,
           at19,at20]
if(index != -1):
    X_test.pop(index)
Xtest = pd.concat(X_test, axis=1)
Xtest = Xtest.as_matrix()

#classifier
rfClf = RandomForestClassifier(n_estimators=100, warm_start=True)
rfClf.fit(X, Y)
test_hat = rfClf.predict(Xtest)
predictedLabels = []
for i in range(0,199):
    if test_hat[i] == 1:
        predictedLabels.append('Good')
    else:
        predictedLabels.append('Bad')
#write results
id = dfTest['Id']
file_name = r'C:\\Users\\user\\PycharmProjects\\ted2\\output\\testSet_Predictions.csv'
newDf = pd.DataFrame({'Client_ID':id,'Predicted_Label':predictedLabels})
newDf.to_csv(file_name, sep='\t')



