import re
from itertools import ifilterfalse
# from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer


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
    
 
         


        
