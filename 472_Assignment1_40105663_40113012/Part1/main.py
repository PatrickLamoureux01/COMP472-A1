import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn as sk
import re
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


APP_FOLDER = r"D:\User\OneDrive - Concordia University - Canada\School\Concordia\Semester_6\COMP472\Assignments\A1\Code\BBC"
totalFiles = 0
x_axis = ['business','entertainment','politics','sport','tech']
y_axis = []
uniqueWords = []

for base, dirs, files in os.walk(APP_FOLDER):
    for Files in files:
        totalFiles += 1
    y_axis.append(totalFiles)
    totalFiles = 0

y_axis.pop(0) # remove the initial 1 from the list that counts the BBC folder
# (2)
plt.bar(x_axis,y_axis,color = 'maroon', width = 0.5)
plt.savefig('BBC-distribution.pdf')
# (3)
res = sk.datasets.load_files(APP_FOLDER,encoding='latin1')
# (4)
count_vect = CountVectorizer()
x = count_vect.fit_transform(res.data)
y = res.target
# (5)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=None)
# (6)
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
predicted = classifier.predict(x_test)
score = classifier.score(x_test, y_test)

# (7)
file = open("bbc-performance.txt", "w")
file.write("(A)" +'\n')
file.write("********************* MultinomialNB default values, try 1 *********************\n")
file.write("(B)" +'\n')
file.write(np.array2string(confusion_matrix(y_test, predicted), separator=', ') +'\n')
file.write("(C)" +'\n')
file.write(classification_report(y_test, predicted, target_names=x_axis) + '\n')
file.write("(D)" +'\n')
file.write("Accuracy: {}".format(accuracy_score(y_test, predicted)) + '\n')
file.write("Macro-Average F1: {}".format(f1_score(y_test, predicted, average='macro')) + '\n')
file.write("Weighted-Average F1: {}".format(f1_score(y_test, predicted, average='weighted')) + '\n')
file.write("(E)" +'\n')
counter = 0
for cl in x_axis:
    file.write("{}: {}".format(cl,y_axis[counter]/sum(y_axis)) + '\n')
    counter += 1
file.write("(F)" +'\n')
file.write("Size of vocabulary: {}".format(len(count_vect.vocabulary_)) + '\n')
file.write("(G)" +'\n')
s = 0
for i in range(5):
    for elem in classifier.feature_count_[i]:
        s += elem
    file.write("Number of word tokens in the {} class: {}".format(x_axis[i],s) + '\n')
    s = 0
file.write("(H)" +'\n')
word_counter = 0
for i in res.data:
    for word in i.split():
        word_counter += 1
file.write("Number of word tokens in the entire corpus: {}".format(word_counter) + '\n')
file.write("(I)" +'\n')
counter = 0
for cl in x_axis:
    file.write("{}: {}".format(cl,np.count_nonzero(classifier.feature_count_[counter])) + '\n')
    counter += 1
file.write("(J)" +'\n')
temp = np.count_nonzero(x.toarray() == 1)
file.write("Number and percentage of words with a frequency of one in the entire corpus: {} - {}%".format(temp, (temp/word_counter) * 100) + '\n')
file.write("(K)" +'\n')
file.write("Word 1 log prob: {}".format(classifier.feature_log_prob_[1][1]) + '\n')
file.write("Word 2 log prob: {}".format(classifier.feature_log_prob_[0][68]) + '\n')
file.write("\n")

# (8)
classifier2 = MultinomialNB()
classifier2.fit(x_train,y_train)
predicted2 = classifier2.predict(x_test)
score = classifier2.score(x_test, y_test)
file = open("bbc-performance.txt", "a")
file.write("(A)" +'\n')
file.write("********************* MultinomialNB default values, try 2 *********************\n")
file.write("(B)" +'\n')
file.write(np.array2string(confusion_matrix(y_test, predicted2), separator=', ') +'\n')
file.write("(C)" +'\n')
file.write(classification_report(y_test, predicted2, target_names=x_axis) + '\n')
file.write("(D)" +'\n')
file.write("Accuracy: {}".format(accuracy_score(y_test, predicted2)) + '\n')
file.write("Macro-Average F1: {}".format(f1_score(y_test, predicted2, average='macro')) + '\n')
file.write("Weighted-Average F1: {}".format(f1_score(y_test, predicted2, average='weighted')) + '\n')
file.write("(E)" +'\n')
counter = 0
for cl in x_axis:
    file.write("{}: {}".format(cl,y_axis[counter]/sum(y_axis)) + '\n')
    counter += 1
file.write("(F)" +'\n')
file.write("Size of vocabulary: {}".format(len(count_vect.vocabulary_)) + '\n')
file.write("(G)" +'\n')
s = 0
for i in range(5):
    for elem in classifier2.feature_count_[i]:
        s += elem
    file.write("Number of word tokens in the {} class: {}".format(x_axis[i],s) + '\n')
    s = 0
file.write("(H)" +'\n')
word_counter = 0
for i in res.data:
    for word in i.split():
        word_counter += 1
file.write("Number of word tokens in the entire corpus: {}".format(word_counter) + '\n')
file.write("(I)" +'\n')
counter = 0
for cl in x_axis:
    file.write("{}: {}".format(cl,np.count_nonzero(classifier2.feature_count_[counter])) + '\n')
    counter += 1
file.write("(J)" +'\n')
temp = np.count_nonzero(x.toarray() == 1)
file.write("Number and percentage of words with a frequency of one in the entire corpus: {} - {}%".format(temp, (temp/word_counter) * 100) + '\n')
file.write("(K)" +'\n')
file.write("Word 1 log prob: {}".format(classifier2.feature_log_prob_[1][1]) + '\n')
file.write("Word 2 log prob: {}".format(classifier2.feature_log_prob_[0][68]) + '\n')
file.write("\n")

# (9)
classifier3 = MultinomialNB(alpha=0.0001)
classifier3.fit(x_train,y_train)
predicted3 = classifier3.predict(x_test)
score = classifier3.score(x_test, y_test)
file = open("bbc-performance.txt", "a")
file.write("(A)" +'\n')
file.write("********************* MultinomialNB default values, try 3 *********************\n")
file.write("(B)" +'\n')
file.write(np.array2string(confusion_matrix(y_test, predicted3), separator=', ') +'\n')
file.write("(C)" +'\n')
file.write(classification_report(y_test, predicted3, target_names=x_axis) + '\n')
file.write("(D)" +'\n')
file.write("Accuracy: {}".format(accuracy_score(y_test, predicted3)) + '\n')
file.write("Macro-Average F1: {}".format(f1_score(y_test, predicted3, average='macro')) + '\n')
file.write("Weighted-Average F1: {}".format(f1_score(y_test, predicted3, average='weighted')) + '\n')
file.write("(E)" +'\n')
counter = 0
for cl in x_axis:
    file.write("{}: {}".format(cl,y_axis[counter]/sum(y_axis)) + '\n')
    counter += 1
file.write("(F)" +'\n')
file.write("Size of vocabulary: {}".format(len(count_vect.vocabulary_)) + '\n')
file.write("(G)" +'\n')
s = 0
for i in range(5):
    for elem in classifier3.feature_count_[i]:
        s += elem
    file.write("Number of word tokens in the {} class: {}".format(x_axis[i],s) + '\n')
    s = 0
file.write("(H)" +'\n')
word_counter = 0
for i in res.data:
    for word in i.split():
        word_counter += 1
file.write("Number of word tokens in the entire corpus: {}".format(word_counter) + '\n')
file.write("(I)" +'\n')
counter = 0
for cl in x_axis:
    file.write("{}: {}".format(cl,np.count_nonzero(classifier3.feature_count_[counter])) + '\n')
    counter += 1
file.write("(J)" +'\n')
temp = np.count_nonzero(x.toarray() == 1)
file.write("Number and percentage of words with a frequency of one in the entire corpus: {} - {}%".format(temp, (temp/word_counter) * 100) + '\n')
file.write("(K)" +'\n')
file.write("Word 1 log prob: {}".format(classifier3.feature_log_prob_[1][1]) + '\n')
file.write("Word 2 log prob: {}".format(classifier3.feature_log_prob_[0][68]) + '\n')
file.write('\n')

# (10)
classifier4 = MultinomialNB(alpha=0.9)
classifier4.fit(x_train,y_train)
predicted4 = classifier4.predict(x_test)
score = classifier4.score(x_test, y_test)
file = open("bbc-performance.txt", "a")
file.write("(A)" +'\n')
file.write("********************* MultinomialNB default values, try 4 *********************\n")
file.write("(B)" +'\n')
file.write(np.array2string(confusion_matrix(y_test, predicted4), separator=', ') +'\n')
file.write("(C)" +'\n')
file.write(classification_report(y_test, predicted4, target_names=x_axis) + '\n')
file.write("(D)" +'\n')
file.write("Accuracy: {}".format(accuracy_score(y_test, predicted4)) + '\n')
file.write("Macro-Average F1: {}".format(f1_score(y_test, predicted4, average='macro')) + '\n')
file.write("Weighted-Average F1: {}".format(f1_score(y_test, predicted4, average='weighted')) + '\n')
file.write("(E)" +'\n')
counter = 0
for cl in x_axis:
    file.write("{}: {}".format(cl,y_axis[counter]/sum(y_axis)) + '\n')
    counter += 1
file.write("(F)" +'\n')
file.write("Size of vocabulary: {}".format(len(count_vect.vocabulary_)) + '\n')
file.write("(G)" +'\n')
s = 0
for i in range(5):
    for elem in classifier3.feature_count_[i]:
        s += elem
    file.write("Number of word tokens in the {} class: {}".format(x_axis[i],s) + '\n')
    s = 0
file.write("(H)" +'\n')
word_counter = 0
for i in res.data:
    for word in i.split():
        word_counter += 1
file.write("Number of word tokens in the entire corpus: {}".format(word_counter) + '\n')
file.write("(I)" +'\n')
counter = 0
for cl in x_axis:
    file.write("{}: {}".format(cl,np.count_nonzero(classifier4.feature_count_[counter])) + '\n')
    counter += 1
file.write("(J)" +'\n')
temp = np.count_nonzero(x.toarray() == 1)
file.write("Number and percentage of words with a frequency of one in the entire corpus: {} - {}%".format(temp, (temp/word_counter) * 100) + '\n')
file.write("(K)" +'\n')
file.write("Word 1 log prob: {}".format(classifier4.feature_log_prob_[1][1]) + '\n')
file.write("Word 2 log prob: {}".format(classifier4.feature_log_prob_[0][68]) + '\n')