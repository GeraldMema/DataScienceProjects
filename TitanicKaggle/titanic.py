#libraries import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats._continuous_distns import halfcauchy_gen
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

#datasets
dataTrainPath = r'C:\\Users\\user\\PycharmProjects\\TitanicWithPython\\dataset\\train.csv'
dataTestPath = 'C:\\Users\\user\\PycharmProjects\\TitanicWithPython\\dataset\\test.csv'
genderSubmission = 'C:\\Users\\user\\PycharmProjects\\TitanicWithPython\\dataset\\gender_submission.csv'
# 
# submit = pd.read_csv(genderSubmission)
titanicTest = pd.read_csv(dataTestPath)

#read train dataset
titanic = pd.read_csv(dataTrainPath)


titanicTest["Class"] = titanicTest.Pclass.map({1: "First", 2: "Second", 3: "Third"})
titanicTest = titanicTest.drop(["Pclass"],axis=1)

surnames = titanicTest.Name.apply(lambda x: x.split(',')[0])#keep only surnames
query = surnames.value_counts().reset_index(name="count").query("count > 1")["index"]#find the names that are not unique
final = surnames.apply(lambda x:query[query.isin([x])].empty)#set true the unique names
titanicTest['FamilyIntitanicTest'] = final.map({True: "Maybe Is Alone", False: "Maybe Has Family"})
titanicTest = titanicTest.drop(["Name"],axis=1)


def getAgeCategory(passenger):
    age= passenger
    if age < 15:
        return "Child"
    elif age <65:
        return "Adult"
    else:
        return "Old"
temp = titanicTest.Age.interpolate()
titanicTest["AgeCategory"] = temp.apply(getAgeCategory)
titanicTest = titanicTest.drop(["Age"],axis=1)

def isAlone(passenger):
    number = passenger
    if number == 0:
        return "Yes"
    else:
        return "No"

titanicTest["IsAlone1"] = titanicTest.SibSp.apply(isAlone)
titanicTest["IsAlone2"] = titanicTest.Parch.apply(isAlone)
titanicTest = titanicTest.drop(["SibSp","Parch"],axis=1)
# sns.factorplot("IsAlone2", data=titanic,kind="count");
# sns.factorplot("IsAlone2", data=titanic,kind="count",hue="Survived");
# sns.factorplot("IsAlone1", data=titanic,kind="count",hue="Sex");
# sns.factorplot("IsAlone2", data=titanic,kind="count",hue="Sex");

#I think that we can do what we did with Name. We will see if we can extract any of information about the dublicate tickets. This
#attribute is difficult to give us any useful information, ticket is a String only and propably a lot of them are diferrents. For
# this reason we have to get another way to get knowlegde from this attribute.
query = titanicTest.Ticket.value_counts().reset_index(name="count").query("count > 1")["index"]#find the tickets that are not unique
final = titanicTest.Ticket.apply(lambda x:query[query.isin([x])].empty)#set true the unique tickets
titanicTest['ExistingTicket'] = final.map({True: "Not Exist", False: "Exists"})
titanicTest = titanicTest.drop(["Ticket"],axis=1)
# sns.factorplot("ExistingTicket", data=titanic,kind="count");
# sns.factorplot("ExistingTicket", data=titanic,kind="count",hue="Survived");
# sns.factorplot("ExistingTicket", data=titanic,kind="count",hue="Sex");

#This attribute is like attribute Age and also in this case we have to estimate which we assume low price tickets
#and high price tickets and in how many bins we have to seperate this. I suppose we may choose 5 bins(very low,low,high,very high)
category_of_prices = ["Very Low","Low","Median","High","Very High"]
titanicTest['PriceOfTicket'] = pd.cut(titanicTest['Fare'],bins=5,labels=category_of_prices)
titanicTest = titanicTest.drop(["Fare"],axis=1)
# sns.factorplot("PriceOfTicket", data=titanic,kind="count");
# sns.factorplot("PriceOfTicket", data=titanic,kind="count",hue="Survived");
# sns.factorplot("PriceOfTicket", data=titanic,kind="count",hue="Sex");

#This is the most difficult attribute for exploring. It could be useful but we have a lot missing data(78%). One good
#preprocessing here is that we can keep only the first letter of Cabin to have less categories for this attribute
#like C,D,E... We can try if this method give us useful information therefore we should fill the missing data with
#a default value that not has prefix a cabin, for example 'X', or we should drop this feature
# Cabin = titanicTest.Cabin.fillna('X')
titanicTest.Cabin = titanicTest.Cabin.astype(str).str[0]
# sns.factorplot("Cabin", data=titanic,kind="count");
# sns.factorplot("Cabin", data=titanic,kind="count",hue="Survived");
# sns.factorplot("Cabin", data=titanic,kind="count",hue="Sex");

#Embarked is clearly categorial attribute that takes 3 values S,C,Q
def getEmbarked(passenger):
    port= passenger
    if port == 'C':
        return "Cherbourg"
    elif port == 'Q':
        return "Queenstown"
    else:
        return "Southampton"
titanicTest["EmbarkedFrom"] = titanicTest.Embarked.apply(getEmbarked)
titanicTest = titanicTest.drop(["Embarked"],axis=1)
# sns.factorplot("EmbarkedFrom", data=titanic,kind="count");
# sns.factorplot("EmbarkedFrom", data=titanic,kind="count",hue="Survived");
# sns.factorplot("EmbarkedFrom", data=titanic,kind="count",hue="Sex");

# #collect data
Pclass = pd.get_dummies(titanicTest.Class)
Name = pd.get_dummies(titanicTest.FamilyIntitanicTest)
Sex = pd.get_dummies(titanicTest.Sex)
Age = pd.get_dummies(titanicTest.AgeCategory)
SibSp = pd.get_dummies(titanicTest.IsAlone1)
Parch = pd.get_dummies(titanicTest.IsAlone2)
Ticket = pd.get_dummies(titanicTest.ExistingTicket)
Fare = pd.get_dummies(titanicTest.PriceOfTicket)
Cabin = pd.get_dummies(titanicTest.Cabin,dummy_na=True)
Embarked = pd.get_dummies(titanicTest.EmbarkedFrom)
print Pclass.shape,Name.shape,Sex.shape,Age.shape,SibSp.shape,Parch.shape,Ticket.shape,Fare.shape,Cabin.shape,Embarked.shape
X_test = [Pclass,Sex,Age,SibSp,Parch,Fare]
XT = pd.concat(X_test,axis=1)
XT = XT.as_matrix()


titanic["Class"] = titanic.Pclass.map({1: "First", 2: "Second", 3: "Third"})
titanic = titanic.drop(["Pclass"],axis=1)
# sns.factorplot("Class", data=titanic,kind="count");
#class with survived
# sns.factorplot("Class", data=titanic,kind="count",hue="Survived");
# sns.factorplot("Class", data=titanic,kind="count",hue="Sex");

#Name is something like passengersID, not very usefull attribute. But if we don't drop this attribute
#we can make some preprocessing and keep only surnames.This is more useful because we can observe if we have matches
# of names, maybe belongs in same family.So, we can seperate in unique and ton unique names. Unique name= maybe has not family inside
# not unique = maybe has family , and the maybe is because to people even has same surname are not in same family
#In first step we keep this attribute and we validate its importance when we use visualization
surnames = titanic.Name.apply(lambda x: x.split(',')[0])#keep only surnames
query = surnames.value_counts().reset_index(name="count").query("count > 1")["index"]#find the names that are not unique
final = surnames.apply(lambda x:query[query.isin([x])].empty)#set true the unique names
titanic['FamilyInTitanic'] = final.map({True: "Maybe Is Alone", False: "Maybe Has Family"})
titanic = titanic.drop(["Name"],axis=1)
# sns.factorplot("FamilyInTitanic", data=titanic,kind="count");
# sns.factorplot("FamilyInTitanic", data=titanic,kind="count",hue="Survived");
# sns.factorplot("FamilyInTitanic", data=titanic,kind="count",hue="Sex");


# Sex is a categorial attribute which takes the values male,female

#Age is a numeric value that is not so important if we keep it as number but we may convert it
#to a categorial like child , adult , old. As categorial value we will see more often the same values
#that means we have more information. Its better the definitions to seperate the ages to be manual from us
#because the pd.cut() has a Normal distribution function depends of minimum and maximum values. So we can tell
# that a passenger under 15 years old is child and up to 65 is old and the others are adults. Our goal is to
#seperate children and old people from others and how survived. Also we have 20% of missing values so i assume that we can
#make an estimation about this value. A good option for numerical data i think is interpolate method of python
def getAgeCategory(passenger):
    age= passenger
    if age < 15:
        return "Child"
    elif age <65:
        return "Adult"
    else:
        return "Old"
temp = titanic.Age.interpolate()
titanic["AgeCategory"] = temp.apply(getAgeCategory)
titanic = titanic.drop(["Age"],axis=1)
# sns.factorplot("AgeCategory", data=titanic,kind="count");
# sns.factorplot("AgeCategory", data=titanic,kind="count",hue="Survived");
# sns.factorplot("AgeCategory", data=titanic,kind="count",hue="Sex");

#On first view we can observe that these two attributes(SibSp,Parch) contains a lot of zero values that means most of passengers
#have zero siblings / spouses /  parents / children aboard the Titanic. So, for these two attributes we can built two
#categories : zero values and non zero values.
def isAlone(passenger):
    number = passenger
    if number == 0:
        return "Yes"
    else:
        return "No"

titanic["IsAlone1"] = titanic.SibSp.apply(isAlone)
titanic["IsAlone2"] = titanic.Parch.apply(isAlone)
titanic = titanic.drop(["SibSp","Parch"],axis=1)
# sns.factorplot("IsAlone2", data=titanic,kind="count");
# sns.factorplot("IsAlone2", data=titanic,kind="count",hue="Survived");
# sns.factorplot("IsAlone1", data=titanic,kind="count",hue="Sex");
# sns.factorplot("IsAlone2", data=titanic,kind="count",hue="Sex");

#I think that we can do what we did with Name. We will see if we can extract any of information about the dublicate tickets. This
#attribute is difficult to give us any useful information, ticket is a String only and propably a lot of them are diferrents. For
# this reason we have to get another way to get knowlegde from this attribute.
query = titanic.Ticket.value_counts().reset_index(name="count").query("count > 1")["index"]#find the tickets that are not unique
final = titanic.Ticket.apply(lambda x:query[query.isin([x])].empty)#set true the unique tickets
titanic['ExistingTicket'] = final.map({True: "Not Exist", False: "Exists"})
titanic = titanic.drop(["Ticket"],axis=1)
# sns.factorplot("ExistingTicket", data=titanic,kind="count");
# sns.factorplot("ExistingTicket", data=titanic,kind="count",hue="Survived");
# sns.factorplot("ExistingTicket", data=titanic,kind="count",hue="Sex");

#This attribute is like attribute Age and also in this case we have to estimate which we assume low price tickets
#and high price tickets and in how many bins we have to seperate this. I suppose we may choose 5 bins(very low,low,high,very high)
category_of_prices = ["Very Low","Low","Median","High","Very High"]
titanic['PriceOfTicket'] = pd.cut(titanic['Fare'],bins=5,labels=category_of_prices)
titanic = titanic.drop(["Fare"],axis=1)
# sns.factorplot("PriceOfTicket", data=titanic,kind="count");
# sns.factorplot("PriceOfTicket", data=titanic,kind="count",hue="Survived");
# sns.factorplot("PriceOfTicket", data=titanic,kind="count",hue="Sex");

#This is the most difficult attribute for exploring. It could be useful but we have a lot missing data(78%). One good
#preprocessing here is that we can keep only the first letter of Cabin to have less categories for this attribute
#like C,D,E... We can try if this method give us useful information therefore we should fill the missing data with
#a default value that not has prefix a cabin, for example 'X', or we should drop this feature
# Cabin = titanic.Cabin.fillna('X')
titanic.Cabin = titanic.Cabin.astype(str).str[0]
# sns.factorplot("Cabin", data=titanic,kind="count");
# sns.factorplot("Cabin", data=titanic,kind="count",hue="Survived");
# sns.factorplot("Cabin", data=titanic,kind="count",hue="Sex");

#Embarked is clearly categorial attribute that takes 3 values S,C,Q
def getEmbarked(passenger):
    port= passenger
    if port == 'C':
        return "Cherbourg"
    elif port == 'Q':
        return "Queenstown"
    else:
        return "Southampton"
titanic["EmbarkedFrom"] = titanic.Embarked.apply(getEmbarked)
titanic = titanic.drop(["Embarked"],axis=1)
# sns.factorplot("EmbarkedFrom", data=titanic,kind="count");
# sns.factorplot("EmbarkedFrom", data=titanic,kind="count",hue="Survived");
# sns.factorplot("EmbarkedFrom", data=titanic,kind="count",hue="Sex");

# #collect data
Pclass = pd.get_dummies(titanic.Class)
Name = pd.get_dummies(titanic.FamilyInTitanic)
Sex = pd.get_dummies(titanic.Sex)
Age = pd.get_dummies(titanic.AgeCategory)
SibSp = pd.get_dummies(titanic.IsAlone1)
Parch = pd.get_dummies(titanic.IsAlone2)
Ticket = pd.get_dummies(titanic.ExistingTicket)
Fare = pd.get_dummies(titanic.PriceOfTicket)
Cabin = pd.get_dummies(titanic.Cabin,dummy_na=True,drop_first=True)
Embarked = pd.get_dummies(titanic.EmbarkedFrom)
print Pclass.shape,Name.shape,Sex.shape,Age.shape,SibSp.shape,Parch.shape,Ticket.shape,Fare.shape,Cabin.shape,Embarked.shape
X_train = [Pclass,Sex,Age,SibSp,Parch,Fare]
X = pd.concat(X_train,axis=1)
X = X.as_matrix()
Y = titanic.Survived

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

param_grid = {[1,5,10,50,100,200,500] }
clf = GridSearchCV(RandomForestClassifier(n_estimators=100, warm_start=True, oob_score = True, random_state = 42), param_grid)


# clf = svm.SVC(kernel='rbf', C = 2.0, gamma='auto')
# clf = RandomForestClassifier(n_estimators=100, warm_start=True)
clf.fit(X,Y)
survived = clf.predict(XT).astype(int)

# write results
# PassengerId = titanicTest.PassengerId;
# survived = float(survived);
# submission = plt.table(PassengerId, survived);
# np.disp(submission(1:5,:))
# writetable(submission,'submission.csv')

#
id = titanicTest['PassengerId']
file_name = 'C:\\Users\\user\\PycharmProjects\\TitanicWithPython\\dataset\\GeraldMema.csv'
newDf = pd.DataFrame({'PassengerId': id, 'Survived': survived})
newDf.to_csv(file_name, index=False)

# #classifiers
# nbClf = MultinomialNB()
# rfClf = RandomForestClassifier(n_estimators=100, warm_start=True)
# svmClf = svm.SVC(kernel='rbf', C = 2.0, gamma='auto')
# knnClf = KNeighborsClassifier(n_neighbors=8)

#
# #cross validation
# numCrossValidation = 20
# crossValidation = StratifiedKFold(Y, n_folds=numCrossValidation)
# #evaluate accuracy
# nbAc = 0
# rfAc = 0
# svmAc = 0
# knnAc = 0
# for i, (train, test) in enumerate(crossValidation):
#     #NB
#     nbClf.fit(X[train], Y[train])
#     nbPred = nbClf.predict(X[test])
#     ac = accuracy_score(Y[test], nbPred)
#     print ac
#     nbAc = nbAc + ac
#     #RF
#     rfClf.fit(X[train], Y[train])
#     rfPred = rfClf.predict(X[test])
#     ac = accuracy_score(Y[test], rfPred)
#     print ac
#     rfAc = rfAc + ac
#     #SVM
#     svmClf.fit(X[train], Y[train])
#     svmPred = svmClf.predict(X[test])
#     ac = accuracy_score(Y[test], svmPred)
#     print ac
#     svmAc = svmAc + ac
#     #KNN
#     knnClf.fit(X[train], Y[train])
#     knnPred = knnClf.predict(X[test])
#     ac = accuracy_score(Y[test], knnPred)
#     print ac
#     knnAc = knnAc + ac
#
# naiveBayesAccuracy = float(nbAc) / numCrossValidation
# randomForestAccuracy = float(rfAc) / numCrossValidation
# supportVectorMachineAccuracy = float(svmAc) / numCrossValidation
# knnAccuracy = float(knnAc) / numCrossValidation
#
# print naiveBayesAccuracy
# print randomForestAccuracy
# print supportVectorMachineAccuracy
# print knnAccuracy
#
# plt.show()
