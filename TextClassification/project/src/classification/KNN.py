from collections import OrderedDict

import numpy
import pandas as pd
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import TruncatedSVD

def knn():
    # Data preparation

    dataTrainPath = 'C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\dataset\\train_set.csv'
    dataTestPath = 'C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\dataset\\test_set.csv'
    df = pd.read_csv(dataTrainPath, sep='\t')

    X_train = df['Content']
    Y_train = df['Category']
    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    Y = le.transform(Y_train)
    # Data Preprocessed
    stop = set(stopwords.words('english'))
    stop.add("said")
    stop.add("say")
    stop.add("will")
    stop.add("new")
    stop.add("also")
    stop.add("one")
    stop.add("now")
    stop.add("still")
    stop.add("time")
    stop.add("may")

    # counter
    count_vect = CountVectorizer(stop_words=stop,min_df=0.03,max_df=0.95)
    X_train = count_vect.fit_transform(X_train)
    # tf
    # count_vect = TfidfVectorizer(stop_words=stop,max_df=0.95,analyzer = 'word')
    # X_train = count_vect.fit_transform(X_train)
    #tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train)
    #dimensionality reduction
    svd_model = TruncatedSVD(n_components=20,random_state=42)
    X_train = svd_model.fit_transform(X_train)

    K = 5

    crossValidation = StratifiedKFold(Y, n_folds=10)
    sumF1 = 0.0
    sumRec = 0.0
    sumPr = 0.0
    sumAc = 0.0
    for i, (train, test) in enumerate(crossValidation):
        predict = []
        for testData in test:
            # First step - calculate the similarities for each testData with trainData
            flagCount = 0
            similaritiesMap = {}
            for trainData in train:
                similarity = numpy.linalg.norm(X_train[trainData]-X_train[testData])
                similaritiesMap.update({similarity:Y[trainData]})
            sortedSimilarities = OrderedDict(sorted(similaritiesMap.items()))
            topK_similarities = dict(sortedSimilarities.items()[:K])
            # Second step - we predict the category with Majority Voting method

            business = 0
            film = 0
            football = 0
            politics = 0
            technology = 0
            for i in range(0,K):
                if topK_similarities.values()[i] == 0:
                    business = business + 1
                if topK_similarities.values()[i] == 1:
                    film = film + 1
                if topK_similarities.values()[i] == 2:
                    football = football + 1
                if topK_similarities.values()[i] == 3:
                    politics = politics + 1
                if topK_similarities.values()[i] == 4:
                    technology = technology + 1
            if max(business,film,football,politics,technology) == business:
                predict.append("Business")
            elif max(business,film,football,politics,technology) == film:
                predict.append("Film")
            elif max(business,film,football,politics,technology) == football:
                predict.append("Football")
            elif max(business,film,football,politics,technology) == politics:
                predict.append("Politics")
            else:
                predict.append("Technology")
        f1 = f1_score(Y_train[test], predict, average="macro")
        sumF1 = sumF1 + f1
        prec = precision_score(Y_train[test],predict, average="macro")
        sumPr = sumPr + prec
        rec = recall_score(Y_train[test], predict, average="macro")
        sumRec = sumRec + rec
        ac = accuracy_score(Y_train[test], predict)
        sumAc = sumAc + ac
    totalScore = []
    totalScore.append(float(sumAc) / 10)
    totalScore.append(float(sumPr)/10)
    totalScore.append(float(sumRec)/10)
    totalScore.append(float(sumF1)/10)
    return totalScore