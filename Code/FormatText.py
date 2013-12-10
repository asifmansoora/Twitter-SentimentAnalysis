import re
from itertools import ifilterfalse
# from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


####################################################################################################################
###         Clean reviews
###
####################################################################################################################


#tokens: List of words
def getPOSList(tokens,Noun=False,Adj = False):
           
        tagged = nltk.pos_tag(tokens)
        #print tagged
        POSList = []
        NounList = []
        AdjList =[]
        # to get all the words that are 'NN','NNP','NNS'
        for i in tagged:
                
                if Noun and i[1][0]=='N':
                        NounList.append(i[0])
                if Adj and i[1][0]=='J':
                        AdjList.append(i[0])
                
                        
        POSList.append(NounList)
        
        POSList.append(AdjList)
#         print "POS List is"
#         for i in POSList:
#             print i
	return POSList


def clean_review(line):
        line = line.lower()
        
       #Convert www.* or https?://* to URL
        line = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',line)
        #print line
        
        #convert n't into 'not'
        line=re.sub('(n\'t)',"not",line)
        
        #handling emoticons
        line=re.sub('(:\))|(:-\))'," happysym",line)
        line=re.sub('(:\()|(:-\()'," sadsym",line)
        
        #Convert @username to AT_USER
        line = re.sub('@[^\s]+','AT_USER',line)
        #print line
        
        #Remove additional white spaces
        line = re.sub('[\s]+', ' ', line)
        #print line
        
        #Replace #word with word
        line = re.sub(r'#([^\s]+)', r'\1', line)
        #print line
        
        #trim
        line = line.strip('\'"')
        #print line
        
        #remove all the characters in string except number, alpha, space and '
        line = re.sub(r'[^a-zA-Z0-9, ,\']', " ", line)
        #print line
        
        tokens = nltk.word_tokenize(line)

        
        return tokens
        
       
#Stemming of the word List by using porter stemmer
# def word_stemming(wordList):
# 
#         
# 	for i,word in enumerate(wordList):
# 		wordList[i] =stem(word)
# 	
# 	return wordList

def filterListofWords(wordList):
        # It is observed that all the words of length less than 2 are not useful.
        # So we are removing all the words that are less than 2
        
        wordList[:] = ifilterfalse(lambda i: (len(i)<3 ) , wordList)
        return wordList

def removeStopwords(wordList):
        #remove the stop words from the NLTK Stop words list
        stopwords = nltk.corpus.stopwords.words('english')
        # adding more stopwords for twitter data
        stopwords.append('URL')
        stopwords.append('AT_USER')
        wordList[:] = ifilterfalse(lambda i: (i in stopwords) , wordList)
        return wordList
 
def removeTwoOrMorechars(wordList):
          
    for i,word in enumerate(wordList):
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        pattern.sub(r"\1\1", word)
        wordList[i]=word
     
    return wordList
       
#Lemmantiation of the word List
def word_Lemmantization(wordList,wordNetLemma):
 
         
	for i,word in enumerate(wordList):
		wordList[i] =wordNetLemma.lemmatize(word)
 	
	return wordList
    
#
def getBagOfWords(corpus):
    vectorizer = CountVectorizer(min_df=2)
    X = vectorizer.fit_transform(corpus)
    print "labels are:"
    print vectorizer.get_feature_names()
    return X.toarray()


def fitCorpus1(corpus):
    vectorizer = CountVectorizer(min_df=1)
    #(ngram_range=(1, 5),analyzer="char", binary=False)
    vectorizer.fit(corpus)
    #print "feature list in unigram:"
    #print vectorizer.get_feature_names()
    return vectorizer

def fitCorpus2(corpus):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1)
    #(ngram_range=(1, 5),analyzer="char", binary=False)
    vectorizer.fit(corpus)
    #print "feature list in bigram:"
    #print vectorizer.get_feature_names()
    return vectorizer


def tranCorpus(corpus,vectorizer1,vectorizer2):
    X1 = vectorizer1.transform(corpus)
    #print "X features in countVect"
    x1 = X1.toarray()
    #print x1
    X2 = vectorizer2.transform(corpus)
    x2 = X2.toarray()
    #print x2
    x1 = np.append(x1,x2,1)
    #print "x1 finally:"
    #print x1
#     j_x2 = 0
#     for rows in x1:
#         rows.extend(x2[j_x2])
#         j_x2 = j_x2 +1
#     print x1
    
    return x1
    
def transform_features_log(x):
    return np.log(1+x)

# To Convert into Scaled data that has zero mean and unit variance:
def transform_features_standardize(x):
    return preprocessing.scale(x) 
         


        
