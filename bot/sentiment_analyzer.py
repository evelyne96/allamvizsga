import os, nltk, string, re, numpy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from .utils import file_control
from . import models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

class SentimentAnalyzer:
    stop_words = set(stopwords.words('english'))

    def __init__(self):
        print("init sentiment analyzer")
        trainD, train_features, self.vectorizer = self.getTrainFeature()
        trainData = [d.sentiment for d in trainD]
        params, score = self.tune_svm_parameters(trainData, train_features)
        print("best score: "+score)
        self.svm_cls = self.svm_classifier(trainData, train_features, params)
        self.nb_cls = self.multinomial_naive_bayes(trainData, train_features)

    def readTrainData(self, model):
        filename = os.path.join(os.getcwd(), 'static', 'sentimentAnalysis', 'train.csv')
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
    def multinomial_naive_bayes(self, trainData, trainFeature):
        #Multinomial naive bayes
        nb = MultinomialNB()
        #setiment = 1 or 0 where 0 = negative, 1 = positive
        nb.fit(trainFeature, trainData)
        #trained naive bayes classifier that can do predictions
        return nb

#Nive Bayes DONE

#SVM
    def svm_classifier(self,trainData, trainFeature, params):
        clf = SVC(C=params['C'], gamma=params['gamma'])
        # clf = svm.SVC()
        clf.fit(trainFeature, trainData)  
        return clf

    def tune_svm_parameters(self, trainData, trainFeature):
        # parameters gamma and C of the Radial Basis Function (RBF) kernel SVM.
        # Intuitively, the gamma parameter defines how far the influence of a single training example
        #  reaches, with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters 
        #  can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.
        # The C parameter trades off misclassification of training examples against simplicity of the decision surface. 
        # A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly 
        # by giving the model freedom to select more samples as support vectors.
        C_range = numpy.logspace(-2, 10, 13)
        gamma_range = numpy.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(trainFeature, trainData)

        return grid.best_params_, grid.best_score_




    
