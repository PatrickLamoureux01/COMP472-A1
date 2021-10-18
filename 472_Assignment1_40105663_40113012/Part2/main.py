import pandas as pandas
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

drug_list = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
my_csv = pandas.read_csv("C:\\Users\\Pub\\Desktop\\AI Mini Project 1\\drug200.csv")
pdf = PdfPages('C:\\Users\\Pub\\Desktop\\AI Mini Project 1\\drug-distribution.pdf')

my_csv.Sex = my_csv.Sex.replace(['F', 'M'], [0, 1])
my_csv.BP = my_csv.BP.replace(['LOW', 'NORMAL', 'HIGH'], [0, 1, 2])
my_csv.Cholesterol = my_csv.Cholesterol.replace(['LOW', 'NORMAL', 'HIGH'], [0, 1, 2])
my_csv.Drug = my_csv.Drug.replace(['drugA', 'drugB', 'drugC', 'drugX', 'drugY'], [0, 1, 2, 3, 4])

inputs = my_csv.drop('Drug', axis='columns')
target = my_csv['Drug']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(inputs, target)

# Use the below two prints statements in the below methods other than plotting
# to see a small subset of what the algorithm predicts from our data
# print(Ytest[:10])
# print(model.predict(Xtest[:10]))

def plotting():
    i = 0
    for drug in drug_list:

        next_drug = my_csv[my_csv.Drug == i]
        plt.plot(next_drug)
        plt.title(drug)

        pdf.savefig()
        plt.clf()
        i += 1
    pdf.close()

def NB_Calc():
    model = GaussianNB()
    model.fit(Xtrain,Ytrain)

    print("==NB_Calc==")
    #print(model.score(Xtest,Ytest))
    #print(confusion_matrix(Ytest, model.predict(Xtest)))
    print(classification_report(Ytest, model.predict(Xtest)))

def BaseDT():
    model = DecisionTreeClassifier()
    model.fit(Xtrain,Ytrain)

    print("==BaseDT==")
    #print(model.score(Xtest,Ytest))
    #print(confusion_matrix(Ytest, model.predict(Xtest)))
    print(classification_report(Ytest, model.predict(Xtest)))

def TopDT():
    clf = GridSearchCV(DecisionTreeClassifier(),{
        'criterion':['gini','entropy'],
        'max_depth':[10,20],
        'min_samples_split':[5,20,40]
    }, cv=5)

    clf.fit(Xtrain,Ytrain)

    print("==TopDT==")
    # print(clf.score(Xtest,Ytest))
    #print(confusion_matrix(Ytest, clf.predict(Xtest)))
    print(classification_report(Ytest, clf.predict(Xtest)))

    #df = pd.DataFrame(clf.cv_results_)
    #print(df)
    #f = open("C:\\Users\\Pub\\Desktop\\AI Mini Project 1\\someData.txt", "a")
    #f.write(df.to_string())
    #f.close()

def PER():
    clf = Perceptron()
    clf.fit(Xtrain,Ytrain)

    print("==PER==")
    # print(clf.score(Xtest, Ytest))
    #print(confusion_matrix(Ytest, clf.predict(Xtest)))
    print(classification_report(Ytest, clf.predict(Xtest)))

def BaseMLP():
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd',)
    clf.fit(Xtrain,Ytrain)

    print("==BaseMLP==")
    # print(clf.score(Xtest, Ytest))
    #print(confusion_matrix(Ytest, clf.predict(Xtest)))
    print(classification_report(Ytest, clf.predict(Xtest)))

def TopMLP():
    clf = GridSearchCV(MLPClassifier(), {
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
        'activation': ['logistic', 'tanh', 'relu', 'identity'],
        'solver': ['adam','sgd']
    }, cv=5)

    clf.fit(Xtrain, Ytrain)

    print("==TopMLP==")
    #print(clf.score(Xtest, Ytest))
    #print(confusion_matrix(Ytest, clf.predict(Xtest)))
    print(classification_report(Ytest, clf.predict(Xtest)))

    #df = pd.DataFrame(clf.cv_results_)
    #print(df)
    #f = open("C:\\Users\\Pub\\Desktop\\AI Mini Project 1\\someData.txt", "a")
    #f.write(df.to_string())
    #f.close()


plotting()

NB_Calc()
BaseDT()
TopDT()
PER()
BaseMLP()
TopMLP()