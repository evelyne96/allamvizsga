import os
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from .utils import file_control
from . import models
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer,TfidfVectorizer
 
class SentimentAnalyzer:
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    trainData = []

    def readTrainData(self, model):
        filename = os.path.join(os.getcwd(), 'bot', 'static', 'sentimentAnalysis', 'train.csv')
        list = file_control.read_from_csv(filename, model)
        return list

    def eliminateStopWords(self):
        traindt = models.TrainData(0,1,'happy')
        trainData = self.readTrainData(traindt)
        all = []
        for data in trainData:
            #since the data is from twitter I will eliminate the user names
            data.sentimentText = re.sub(r'@([A-Za-z0-9_]+)', '', data.sentimentText)
            all.append(data.sentimentText)
            # data.sentimentText = data.sentimentText.translate(str.maketrans('','',string.punctuation))
            # data.words = word_tokenize(data.sentimentText)
            # for r in data.words:
            #     if r in self.stop_words:
            #         data.words.remove(r)
            # all.append(data.words)
        
        #this does the tokenizing as well and eliminates the stop words it also ignores puntuations
        sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False,
                                         sublinear_tf=True, stop_words=self.stop_words)
        sklearn_representation = sklearn_tfidf.fit_transform(all)

        #file_control.write_to("vocabulary.txt",sklearn_tfidf.vocabulary_)
        return trainData
        





    
