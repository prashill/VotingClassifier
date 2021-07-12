# -*- coding: utf-8 -*-
"""


@author: prashil
"""

## PRASHIL SONI – S00375453
## SPRING 2021 - Machine Learning, HW_2

# Loading required modules
from sklearn.datasets import load_digits              # Import digits dataset
from sklearn.neighbors import KNeighborsClassifier    # Import KNeighbors Classifier
from sklearn.naive_bayes import GaussianNB            # Import naive_bayes Classifier
from sklearn.tree import DecisionTreeClassifier       # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier   # Import Random Forest Classifier
from sklearn import svm                               # Import Support Vector Machines Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading wine dataset
digits_dataset = load_digits()

# printing description of dataset
# print(digits_dataset.DESCR) 

##############################################################some testing

# X_train = X_train[y==7]
# x = digits_dataset['data']
# y = digits_dataset['target']
# x = x[y==4]
# y  = y[y==4]


# X_train, X_test, y_train, y_test = train_test_split(
#     x,y, test_size=0.50,  random_state=0)
##############################################################

# use train test split to split data
X_train, X_test, y_train, y_test = train_test_split(
    digits_dataset['data'],digits_dataset['target'], test_size=0.50,  random_state=0)


# just checking a data
# print("X train", X_train) 
# print("Y train", y_train)
# print("X test", X_test)
# print("y test", y_test)

# KNN algoritm Implementation
knn = KNeighborsClassifier(n_neighbors=1) 
knn.fit(X_train, y_train)        # Fit the classifier to the data
predKnn = knn.predict(X_test) # predicting a data using KNN
# print(pred_knn)
# print(len(pred_knn)) # checking length 

# Naive Bayes Implementation
nb = GaussianNB()
nb.fit(X_train, y_train)       # Fit the classifier to the data
predNb = nb.predict(X_test) # predicting a data using naive bayes
# print(pred_nb)
# print(len(pred_nb)) # checking length

# Decision Trees Implementation
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)       # Fit the classifier to the data
predDt = dt.predict(X_test) # predicting a data using Decision Trees
#print(len(pred_dt)) # checking length

# Random Forests Implementation
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)       # Fit the classifier to the data
predRf = rf.predict(X_test) # predicting a data using Random Forests
# print(len(pred_rf)) # checking length

# Support Vector Machines Implementation
svmc = svm.SVC(kernel='rbf')
svmc.fit(X_train, y_train)         # Fit the classifier to the data
predSvmc = svmc.predict(X_test) # predicting a data using Support Vector Machines

# Voting classifier logic: what I did is make a loop that will go to all of the 5 classifier's predicted data
# and will take out most voted number from predicted data than insert all of this data to one list.

votingMajor = []
testList = []

from collections import Counter

for n in range(len(predKnn)):
    myList = []
    myTest = []
    
    myTest.append(y_test[n])
    myList.append(predKnn[n])  
    myList.append(predNb[n])
    myList.append(predDt[n])
    myList.append(predRf[n])
    myList.append(predSvmc[n])
    data = Counter(myList)
    
# print(data.most_common(1)[0][0])
    votingMajor.append(data.most_common(1)[0][0])
    testList.append(myTest)
    
#print(votingMajor) #just checking voting data
# print(testList)

# getting testing accuracy
print("testing accuracy of KNN is :{:2f} ".format(knn.score(X_test,y_test)))
print("testing accuracy of Naive Bayes is :{:2f} ".format(nb.score(X_test,y_test)))
print("testing accuracy of Decision Trees is :{:2f} ".format(dt.score(X_test,y_test)))
print("testing accuracy of Random Forests is :{:2f} ".format(rf.score(X_test,y_test)))
print("testing accuracy of Support Vector Machines is :{:4f} ".format(svmc.score(X_test,y_test))) 
print("testing accuracy of Majority Voting is: ",round(accuracy_score(y_test, votingMajor),2))

zero = []
testZero = []

for i in range(len(votingMajor)):
    if votingMajor[i] == 0:
        zero.append(votingMajor[i])
        testZero.append(testList[i])
print("Test Accuracy of 0 is: ",accuracy_score(testZero, zero))


one = []
testOne = []
for i in range(len(votingMajor)):
    if votingMajor[i] == 1:
        one.append(votingMajor[i])
        testOne.append(testList[i])
# print(testOne) 
print("Test Accuracy of 1 is: ",round(accuracy_score(testOne, one),2))


two = []
testTwo = []
for i in range(len(votingMajor)):
    if votingMajor[i] == 2:
        two.append(votingMajor[i])
        testTwo.append(testList[i])
# print(testTwo) 
print("Test Accuracy of 2 is: ", accuracy_score(testTwo, two))

three = []
testThree = []
for i in range(len(votingMajor)):
    if votingMajor[i] == 3:
        three.append(votingMajor[i])
        testThree.append(testList[i])
# print(testThree) 
print("Test Accuracy of 3 is: ",round(accuracy_score(testThree, three),2))

four = []
testFour = []
for i in range(len(votingMajor)):
    if votingMajor[i] == 4:
        four.append(votingMajor[i])
        testFour.append(testList[i])
# print(testFour) 
print("Test Accuracy of 4 is: ",round(accuracy_score(testFour, four),2))

five = []
testFive = []
for i in range(len(votingMajor)):
    if votingMajor[i] == 5:
        five.append(votingMajor[i])
        testFive.append(testList[i])
# print(testFive) 
print("Test Accuracy of 5 is: ",round(accuracy_score(testFive, five),2))

six = []
testSix = []
for i in range(len(votingMajor)):
    if votingMajor[i] == 6:
        six.append(votingMajor[i])
        testSix.append(testList[i])
# print(testSix) 
print("Test Accuracy of 6 is: ",round(accuracy_score(testSix, six),2))

seven = []
testSeven = []
for i in range(len(votingMajor)):
    if votingMajor[i] == 7:
        seven.append(votingMajor[i])
        testSeven.append(testList[i])
# print(testSeven) 
print("Test Accuracy of 7 is: ",round(accuracy_score(testSeven, seven),2))

eight = []
testEight = []
for i in range(len(votingMajor)):
    if votingMajor[i] == 8:
        eight.append(votingMajor[i])
        testEight.append(testList[i])
# print(testEight) 
print("Test Accuracy of 8 is: ",round(accuracy_score(testEight, eight),2))

nine = []
testNine = []
for i in range(len(votingMajor)):
    if votingMajor[i] == 9:
        nine.append(votingMajor[i])
        testNine.append(testList[i])
# print(testNine) 
print("Test Accuracy of 9 is: ", round(accuracy_score(testNine, nine),2))



