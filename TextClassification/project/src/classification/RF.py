import pandas as pd
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

def rf():
    # Data preparation
    dataTrainPath = 'C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\dataset\\train_set.csv'
    dataTestPath = 'C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\dataset\\test_set.csv'
    df = pd.read_csv(dataTrainPath, sep='\t')

    X_train = df['Content']
    Y_train = df['Category']


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
    le = preprocessing.LabelEncoder()
    le.fit(df["Category"])
    Y = le.transform(df["Category"])
    # tf
    # count_vect = TfidfVectorizer(stop_words=stop,max_df=0.95,analyzer = 'word')
    # X_train = count_vect.fit_transform(X_train)
    #tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train)
    #dimensionality reduction
    svd_model = TruncatedSVD(n_components=20,random_state=42)
    X_train = svd_model.fit_transform(X_train)
    #classifier
    clf = RandomForestClassifier(n_estimators=100, warm_start=True)

    crossValidation = StratifiedKFold(Y, n_folds=10)
    sumF1 = 0.0
    sumRec = 0.0
    sumPr = 0.0
    sumAc = 0.0
    for i, (train, test) in enumerate(crossValidation):
        clf.fit(X_train[train], Y_train[train])
        test_hat = clf.predict(X_train[test])
        f1 = f1_score(Y_train[test], test_hat, average="macro")
        sumF1 = sumF1 + f1
        prec = precision_score(Y_train[test], test_hat, average="macro")
        sumPr = sumPr + prec
        rec = recall_score(Y_train[test], test_hat, average="macro")
        sumRec = sumRec + rec
        ac = accuracy_score(Y_train[test], test_hat)
        sumAc = sumAc + ac
    totalScore = []
    totalScore.append(float(sumAc) / 10)
    totalScore.append(float(sumPr) / 10)
    totalScore.append(float(sumRec) / 10)
    totalScore.append(float(sumF1) / 10)

    return totalScore
