'''
Created on Nov 2, 2013

@author: asifmansoor
'''


import time
start_time = time.time()

from sklearn import linear_model
from sklearn.metrics import auc_score
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcess
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors
# Confusion_matrix
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer
from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import KFold
from sklearn import cross_validation



from nltk.collocations import* 

from FormatText import * #external function file
from ModelFunctions import *
from FormatFeatures import *

# import the text analysis later for other useful features
#from Text_Analysis import *
from logger import *

from sklearn.cross_validation import KFold

def RunCrossValidation2(X, Y, modelName,k):
	listOfAccrate = []
	kf = KFold(len(Y), k, shuffle =True)
	for train_index, test_index in kf:
		X_train,X_test = X[train_index],X[test_index]
		y_train,y_test = Y[train_index],Y[test_index]
		modelName.fit(X_train,y_train)
		y_predict = modelName.predict(X_test)
		cm = confusion_matrix(y_test, y_predict)
		cm_num = np.array(cm)
		print "Confusion Matrix: "
		print cm_num
		print "sum of diagonal =",+np.trace(cm)
		print "sum of all CM =",+np.sum(cm)
 
		print "Classifier Accuracy ="
		
		accRate = float((float(np.trace(cm))/float(np.sum(cm)))) * 100
		print accRate
		listOfAccrate.append(accRate)

	listOfAccrate = np.array(listOfAccrate)
	return np.mean(listOfAccrate)
	

#Preprocessing data

#################################################################################
# MODEL PARAMETERS
#################################################################################

#bCrossValidation = False
#kFold = 5
#bRunLiveYelpTest = True

sTrainingFile = 'Train-Human_codedCessationTweetsTilta'
#sTestingFile = 'consolidated_yelp_testing_data.csv'
#sValidatinFile = 'consolidated_yelp_validation_data.csv'

#Classifer_Name = 'svr'



wordNetLemma = WordNetLemmatizer()

#IgnoreColumn = 'total_reviews_written_by_user'

#################################################################################
# LOADING TRAINING DATA
#################################################################################
 
trainfile = open(sTrainingFile,"rU")
header = trainfile.next().rstrip().split('~@@')

print header

y = []
X = []

reviewData = []

corpus=[]

iCount = 0 # calculates the number of sample 

print "\t Time is: "+str(time.time()-start_time)
print "LOADING THE TRAINING DATA"



for line in trainfile:
    #print line
    reviewData = []
    splitted = line.rstrip().split('~@@')
    
    #print splitted[2]
    #   Processing Tweets   #
    wordList =  clean_review(splitted[2])   #To clean all the nonAlphaNumeric character
    
#     print "tokens from the line after clean and token functions: "
#     for words in wordList:
#        print words  
    
    
    wordList = filterListofWords(wordList)  #to Filter words less than length 3
#     print "tokens from the line after removing word less than 2: "
#     for words in wordList:
#        print words  
    
    wordList = removeStopwords(wordList)     #remove stop words
#     print "tokens from the line after removing stop words: "
#     for words in wordList:
#        print words  
    

    
    splitted[1] = len(wordList) #adding the feature, as length of document
    
#     wordList =  word_stemming(wordList) #I think Below function word_Lemmantization will give good result than stemming
    wordList = word_Lemmantization(wordList,wordNetLemma)
#    print wordList
#    print "tokens from the line after Stemming: "
#    for words in wordList:
#       print words 
    
#    print "join the tokens and add as line"
    corpus.append(" ".join(wordList))
#    print corpus
    
    
    
    reviewData.append(len(wordList))
#     print "content of reviewData before POS"
#     for i in reviewData:
#         print i
    
    wordList =  getPOSList(wordList,Noun=True)
#     print "length of word list is,"
#     print len(wordList[0])
    reviewData.append(len(wordList[0])) # getting the list of nouns
#     print "content of reviewData after POS"
#     for i in reviewData:
#         print i
    
    # Adding label as numerical value
    
    if (splitted[4] =='Cessation'):
        y.append(1)
    else:
        y.append(0)
    
    X.append(reviewData)

#    if(iCount == 100):
#        break  
    
   # X_train.append(features)
    if(iCount % 100==0):
        print "Processed",+iCount
                  
    iCount = iCount +1
    
trainfile.close()

print corpus
bagofwrd = getBagOfWords(corpus)
print bagofwrd

#print reviewData
print X
print y
print "\t Data loaded succesfull: # of rows: ",len(X), " # of Columns: ",len(X[0])," Length of labels: ",len(y)
#
print "\t Time is: "+str(time.time()-start_time)+"\n"





# Split the data into a training set and a test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# X_train, X_test, y_train, y_test = train_test_split(bagofwrd, y, random_state=42)
# 
# 
# X_train= np.array(X_train)
# X_test= np.array(X_test)
# 
# y_train= np.array(y_train)
# y_test= np.array(y_test)
# 
# print "X_train : "
# print X_train
# print len(X_train)
# 
# print "y_train : "
# print y_train
# print len(y_train)
# 
# print "X_test : "
# print len(X_test)
# 
# print "y_test : "
# print len(y_test)

# Run classifier
classifier = svm.SVC(kernel='linear')

gnb = BernoulliNB();

x= np.array(bagofwrd)
y= np.array(y)


np_mean =RunCrossValidation2(x, y, gnb,10)
print "Accuracy means (NB) ="
print np_mean

print "*******************"
np_mean =RunCrossValidation2(x, y, classifier,10)
print "Accuracy means (SVM)="
print np_mean

#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)

#cm = confusion_matrix(y_test, y_pred)
# print(cm)
#  
# cm_num = np.array(cm)
# print "sum of diagonal =",+np.trace(cm)
# print "sum of all CM =",+np.sum(cm)
#  
# print "Classifier Accuracy ="
# print float((float(np.trace(cm))/float(np.sum(cm)))) * 100
# scores = cross_validation.cross_val_score(classifier,bagofwrd,y,cv=5)
# 
# print scores


	








