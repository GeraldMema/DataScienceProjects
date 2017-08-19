import csv

#my classifier - output  testSet_categories.csv
execfile("C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\src\\classification\\MINE.py")
myClassifier = mine()

#k-means - output clustering_KMeans.csv
execfile("C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\src\\clustering\\K-means.py")
kmeans=kMeans()


#output EvaluationMetric_10fold.csv
result = [['Statistic Measure','Naive Bayes','Random Forest','SVM','KNN']]

execfile("C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\src\\classification\\NB.py")
nb = naiveBayes()

execfile("C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\src\\classification\\RF.py")
rf = rf()

execfile("C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\src\\classification\\SVM.py")
svm = support_vector_machine()

execfile("C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\src\\classification\\KNN.py")
knn = knn()

accuracy = ['Accuracy']
accuracy.append(nb[0])
accuracy.append(rf[0])
accuracy.append(svm[0])
accuracy.append(knn[0])
result.append(accuracy)

precision = ['Precision']
precision.append(nb[1])
precision.append(rf[1])
precision.append(svm[1])
precision.append(knn[1])
result.append(precision)

recall = ['Recall']
recall.append(nb[2])
recall.append(rf[2])
recall.append(svm[2])
recall.append(knn[2])
result.append(recall)

f1_score = ['F-Measure']
f1_score.append(nb[3])
f1_score.append(rf[3])
f1_score.append(svm[3])
f1_score.append(knn[3])
result.append(f1_score)

# write the results in csv file
with open('C:\\Users\\user\\Documents\\Mine\\Projects\\psilos\\project\\outputFiles\\EvaluationMetric_10fold.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(result)








