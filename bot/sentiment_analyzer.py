import os
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from .utils import file_control
from . import models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

class SentimentAnalyzer:
    stop_words = set(stopwords.words('english'))

    def __init__(self):
        print("init sentiment analyzer")
        trainD, self.train_features, self.vectorizer = self.getTrainFeature()
        self.trainData = [d.sentiment for d in trainD]
        self.svm_cls = self.svm_classifier()
        self.nb_cls = self.multinomial_naive_bayes()

    def readTrainData(self, model):
        filename = os.path.join(os.getcwd(), 'bot', 'static', 'sentimentAnalysis', 'train.csv')
        list = file_control.read_from_csv(filename, model)
        return list

#preproceccing data
    def eliminateStopWords(self,trainData):
        for data in trainData:
            #since the data is from twitter I will eliminate the user names
            data.sentimentText = re.sub(r'@([A-Za-z0-9_]+)', '', data.sentimentText)
            #replace #word with word
            data.sentimentText = re.sub(r'#([^\s]+)', r'\1', data.sentimentText)
            #Remove not alphanumeric symbols white spaces
            data.sentimentText = re.sub(r'[^\w]', ' ', data.sentimentText)

        return trainData

    #stemming!!!
    def tokenize(self,text):
        stemmer = SnowballStemmer("english")
        stems = [stemmer.stem(t) for t in word_tokenize(text)]
        return stems
        
#preproceccing data DONE
    def getTrainFeature(self):
        traindt = models.TrainData(0,1,'happy')
        trainData = self.readTrainData(traindt)
        #eliminate the user names since the stop words will be eliminated by scikit tfidfrepresentation
        trainData = self.eliminateStopWords(trainData)

    #     #this does the tokenizing as well and eliminates the stop words it also ignores puntuations
    #     #so basically we will get the vocabulary of every word used except the stop words
    #     #this is a dictionary of word:id pair
        vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False,
                                    sublinear_tf=True, stop_words=self.stop_words, tokenizer=self.tokenize,
                                    analyzer='word')
    #     #this will give us the tf * idf weight for every word in our documents in (documentNR, wordID) -> tf_idf pairs
        train_features = vectorizer.fit_transform([d.sentimentText for d in trainData])
        return trainData, train_features, vectorizer

#Naive Bayes.
    def multinomial_naive_bayes(self):
        #Multinomial naive bayes
        nb = MultinomialNB()
        #setiment = 1 or 0 where 0 = negative, 1 = positive
        nb.fit(self.train_features, self.trainData)
        #trained naive bayes classifier that can do predictions
        return nb

#Nive Bayes DONE

#SVM
    def svm_classifier(self):
        clf = svm.LinearSVC()
        clf.fit(self.train_features, self.trainData)  
        return clf




    
