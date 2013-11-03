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
from sklearn import neighbors
# Confusion_matrix
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer
from sklearn.cross_validation import train_test_split



from nltk.collocations import* 

from FormatText import * #external function file
from ModelFunctions import *
from FormatFeatures import *

# import the text analysis later for other useful features
#from Text_Analysis import *
from logger import *

from sklearn.cross_validation import KFold

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
#     print "tokens from the line after Stemming: "
#     for words in wordList:
#       print words 
    
    
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

#     if(iCount == 1000):
#         break  
    
   # X_train.append(features)
    if(iCount % 100==0):
        print "Processed",+iCount
                  
    iCount = iCount +1
    
trainfile.close()

#print reviewData
print X
print y
print "\t Data loaded succesfull: # of rows: ",len(X), " # of Columns: ",len(X[0])," Length of labels: ",len(y)
#
print "\t Time is: "+str(time.time()-start_time)+"\n"


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train= np.array(X_train)
X_test= np.array(X_test)

y_train= np.array(y_train)
y_test= np.array(y_test)

print "X_train : "
print X_train
print len(X_train)

print "y_train : "
print y_train
print len(y_train)

print "X_test : "
print len(X_test)

print "y_test : "
print len(y_test)

# Run classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
 
cm = confusion_matrix(y_test, y_pred)
print(cm)

cm_num = np.array(cm)
print "sum of diagonal =",+np.trace(cm)
print "sum of all CM =",+np.sum(cm)

print "Classifier Accuracy ="
print float((float(np.trace(cm))/float(np.sum(cm)))) * 100







