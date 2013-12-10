'''
Created on Dec 3, 2013

@author: asifmansoor
'''

from __future__ import division
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

from sklearn.ensemble import RandomForestClassifier


from collections import OrderedDict



from nltk.collocations import* 

from FormatText import * #external function file
from ModelFunctions import *
from FormatFeatures import *

# import the text analysis later for other useful features
#from Text_Analysis import *
from logger import *

def ClassifierAccuracy(modelName,X_test,y_test):
    y_predict = modelName.predict(X_test)
    #result_log.write(str("Y_prediction: "+str(y_predict)+"\n"))
    #print "Y_Actual:"
    #PrintActY(y_test)
    
    #print "Y_Predict:"
    #PrintY(y_predict)

    cm = confusion_matrix(y_test, y_predict)
    cm_num = np.array(cm)
    print "Confusion Matrix: "+str(cm_num)
    final_log.write(str("Confusion Matrix: "+str(cm_num)+"\n"))
    
    #print "sum of diagonal = "+str(np.trace(cm))
    #result_log.write(str("sum of all CM = "+str(np.sum(cm))+"\n"))
    accRate = float((float(np.trace(cm))/float(np.sum(cm)))) * 100
    print "Classifier Accuracy = " +str(accRate)
    final_log.write(str("Classifier Accuracy = "+str(accRate)+"\n"))
    
    pr_cess = float(cm[1,1] / (cm[1,1] + cm[0,1]))
    print "Precision for Cessation Data: "+ str(pr_cess)
    final_log.write(str("Precision for Cessation Data: "+str(pr_cess)+"\n"))
    
    re_cess = float(cm[1,1] / (cm[1,1] + cm[1,0]))
    print "Recall for Cessation Data: "+ str(re_cess)
    final_log.write(str("Recall for Cessation Data: "+str(re_cess)+"\n"))
    
    f_cess = float((2*pr_cess*re_cess) / (pr_cess+re_cess))
    print "F-Score for Cessation Data: " + str(f_cess)
    final_log.write(str("F-Score for Cessation Data: "+str(f_cess)+"\n"))
    
    pr_nocess = float(cm[0,0] / (cm[0,0] + cm[1,0]))
    print "Precision for No Cessation Data: "+ str(pr_nocess)
    final_log.write(str("Precision for No Cessation Data: "+str(pr_nocess)+"\n"))
    
    re_nocess = float(cm[0,0] / (cm[0,0] + cm[0,1]))
    print "Recall for No Cessation Data: "+ str(re_nocess)
    final_log.write(str("Recall for No Cessation Data: "+str(re_nocess)+"\n"))
    
    f_nocess = float((2*pr_nocess*re_nocess) / (pr_nocess+re_nocess))
    print "F-Score for No Cessation Data: " + str(f_nocess)
    final_log.write(str("F-Score for No Cessation Data: "+str(f_nocess)+"\n"))
    

def PrintY(Y_pred):
    cessation_index =0
    nocessation_index =0
    for i in Y_pred:
        print i
        if int(i) == 1:
            cessation_index = cessation_index +1
        else:
            nocessation_index = nocessation_index +1
    print "Predicted Cessation Numbers " +str(cessation_index)
    print "Predicted No Cessation Numbers " +str(nocessation_index)

def PrintActY(Y_pred):
    cessation_index =0
    nocessation_index =0
    for i in Y_pred:
        print i
        if int(i) == 1:
            cessation_index = cessation_index +1
        else:
            nocessation_index = nocessation_index +1
    print "Actual Cessation Numbers " +str(cessation_index)
    print "Actual No Cessation Numbers " +str(nocessation_index)

final_log = open("final_log.csv","a")

    
sTrainingFile = 'Train-Human_codedCessationTweetsTilta'

wordNetLemma = WordNetLemmatizer()

trainfile = open(sTrainingFile,"rU")
header = trainfile.next().rstrip().split('~@@')

print header

y = []
X = []

reviewData = []

corpus=[]

iCount = 0 # calculates the number of sample 
final_log.write(str("***************log*************************"+"\n"))
final_log.write(str("Time logged :"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n"))
print "\t Time is: "+str(time.time()-start_time)
print "LOADING THE TRAINING DATA"

for line in trainfile:
    #print line
    reviewData = []
    splitted = line.rstrip().split('~@@')
    
    #   Processing Tweets   #
    wordList =  clean_review(splitted[2])   #To clean all the nonAlphaNumeric character
    wordList = filterListofWords(wordList)  #to Filter words less than length 3
    wordList = removeStopwords(wordList)     #remove stop words
    wordList = removeTwoOrMorechars(wordList)
    
    splitted[1] = len(wordList) #adding the feature, as length of document
    wordList = word_Lemmantization(wordList,wordNetLemma)
    
    corpus.append(" ".join(wordList))
    
    reviewData.append(len(wordList))
    
    wordList =  getPOSList(wordList,Noun=True)
    reviewData.append(len(wordList[0])) 
    
    
        # Adding label as numerical value
    
    if (splitted[4] =='Cessation'):
        y.append(1)
    else:
        y.append(0)
    
    X.append(reviewData)
    
#     if(iCount == 2000):
#         break  
    if(iCount % 100==0):
        print "Processed",+iCount
                  
    iCount = iCount +1
    
trainfile.close()
final_log.write(str("Training Data Processed"+"\n"))

unique_corpus = []
y_index = 0
y_index_list = []
# 
for line in corpus:
    if line not in unique_corpus:
        unique_corpus.append(line)
    else:
        y_index_list.append(y_index)
    y_index =  y_index+1


if unique_corpus is None:
    final_log.write(str("Corpus with Duplicates (Retweets)"+"\n"))
else:
    final_log.write(str("Corpus without Duplicates (Retweets)"+"\n"))
    
#print "Length of UNique of Corpus " + str(len(unique_corpus) )
y = [i for j, i in enumerate(y) if j not in y_index_list]

# unique_corpus = corpus

y_train= np.array(y)

# Transform into features
# Convert to unigram
vect1 = fitCorpus1(unique_corpus)
final_log.write(str("Unigrams Added"+"\n"))
# Convert to bigram
vect2 = fitCorpus2(unique_corpus)
final_log.write(str("Bigrams Added"+"\n"))

X_train = tranCorpus(unique_corpus,vect1,vect2)

X_train = transform_features_log(X_train)

# Run classifier
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(X_train,y_train)
print "SVM Model Has been learned"
final_log.write(str("SVM Linear Model learned"+"\n"))


# 
# gnb = BernoulliNB();
# gnb.fit(X_train,y_train)
# print "NB Model Has been learned"
# final_log.write(str("NB - Bernoulli Model learned"+"\n"))
# 
# # 
# # x= np.array(bagofwrd)
# 
# 
# clf_rf = RandomForestClassifier(n_estimators=25)
# clf_rf.fit(X_train,y_train)
# print "Random forest Model Has been learned"
# final_log.write(str("Random forest Model (tree = 25) learned"+"\n"))
# 
# clf_maxEnt = linear_model.LogisticRegression(tol=1e-8, penalty='l2', C=20)
# clf_maxEnt.fit(X_train,y_train)
# print "Maximum Entropy Model Has been learned"
# final_log.write(str("Maximum Entropy Model learned"+"\n"))


print "*******************"
print "All Model Has been learned"



# Testing data Processing

sTestingFile = 'Test-Human_codedCessationTweetsTilta.txt'

wordNetLemma = WordNetLemmatizer()

testfile = open(sTestingFile,"rU")
test_header = testfile.next().rstrip().split('~@@')

print test_header

y = []
X = []

reviewData = []

corpus=[]

iCount = 0 # calculates the number of sample 

print "\t Time is: "+str(time.time()-start_time)
print "LOADING THE TESTING DATA"

for line in testfile:
    #print line
    reviewData = []
    splitted = line.rstrip().split('~@@')
    
    #   Processing Tweets   #
    wordList =  clean_review(splitted[2])   #To clean all the nonAlphaNumeric character
    wordList = filterListofWords(wordList)  #to Filter words less than length 3
    wordList = removeStopwords(wordList)     #remove stop words
    wordList = removeTwoOrMorechars(wordList)
    
    splitted[1] = len(wordList) #adding the feature, as length of document
    wordList = word_Lemmantization(wordList,wordNetLemma)
    
    corpus.append(" ".join(wordList))
    
    reviewData.append(len(wordList))
    
    wordList =  getPOSList(wordList,Noun=True)
    reviewData.append(len(wordList[0])) 
    
    
        # Adding label as numerical value
    
    if (splitted[4] =='Cessation'):
        y.append(1)
    else:
        y.append(0)
    
    X.append(reviewData)
    
    if(iCount % 100==0):
        print "Processed",+iCount
                  
    iCount = iCount +1
    
testfile.close()
final_log.write(str("Testing Data Processed"+"\n"))

y_test= np.array(y)

X_test = tranCorpus(corpus,vect1,vect2)
X_test = transform_features_log(X_test)
final_log.write(str("Testing Data has been transformed to Training data parameters"+"\n"))

print "Classifier result for SVM"
final_log.write(str("Classifier result for SVM"+"\n"))
ClassifierAccuracy(clf_svm,X_test,y_test)
final_log.write(str("******************************"+"\n"))

# print "Classifier result for NB"
# final_log.write(str("Classifier result for NB"+"\n"))
# ClassifierAccuracy(gnb,X_test,y_test)
# final_log.write(str("******************************"+"\n"))
# 
# print "Classifier result for Random Forest"
# final_log.write(str("Classifier result for Random Forest"+"\n"))
# ClassifierAccuracy(clf_rf,X_test,y_test)
# final_log.write(str("******************************"+"\n"))
# 
# print "Classifier result for Logistic Regression "
# final_log.write(str("Classifier result for Max Entropy"+"\n"))
# ClassifierAccuracy(clf_maxEnt,X_test,y_test)
# final_log.write(str("******************Completed End of File ************"+"\n"))

