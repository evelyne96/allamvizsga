import os, nltk, string, re, numpy, csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from numpy import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
 
import matplotlib.pyplot as plt  
from scipy.io import loadmat  

def read_from_csv(filename):
    modelList = []
    if os.path.exists(filename):
        with open(filename) as file:
            reader = csv.reader(file)
            for row in reader:
                #sentiment-0,1 and sentimenttext
                modelList.append((row[1],row[2]))
    return modelList

def readTrainData():
        filename = os.path.join(os.getcwd(), 'static', 'sentimentAnalysis', 'train.csv')
        list = read_from_csv(filename)
        print(len(list))
        list = list[1:]
        random.shuffle(list)
        print(len(list))
        test = list[90000:95000]
        print(len(test))
        list = list[:90000]
        print(len(list))
        return list, test

sstop_words = set(stopwords.words('english'))

def classify():
        print("init sentiment analyzer")
        trainD, train_features, vectorizer, test = getTrainFeature()
        trainData = [sentiment for (sentiment,sentimentText) in trainD]
        params = dict( gamma = 0.0035111917342151274, C = 3162.2776601683795)
        #params, score = tune_svm_parameters(trainData, train_features)       
        #print("best score: "+str(score)+" best params: "+str(params))
        svm_cls = svm_classifier(trainData, train_features, params)
        print('SVC')
        accuracy(svm_cls, test, vectorizer)
        nb_cls = multinomial_naive_bayes(trainData, train_features)
        print('Naive bayes')
        accuracy(nb_cls, test, vectorizer)
        # plot_test2D(train_features, trainData)
        # plot_test3D(train_features, trainData)
        #visualize_parameters(train_features, trainData, params)
        # pickle.dump(svm_cls, open('svm_classifier.pkl', 'wb'))
        # pickle.dump(nb_cls, open('nb_classifier.pkl', 'wb'))
        # pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
		
def accuracy(classifier, testData, vectorizer):
       data = vectorizer.transform([text for (s, text) in testData])
       p = classifier.predict(data)
       print('accuracy', accuracy_score([s for (s, text) in testData], p))

#preproceccing data
def eliminateStopWords(trainData):
        for (sentiment,sentimentText) in trainData:
            #since the data is from twitter I will eliminate the user names
            sentimentText = re.sub(r'@([A-Za-z0-9_]+)', '', sentimentText)
            #replace #word with word
            sentimentText = re.sub(r'#([^\s]+)', r'\1', sentimentText)
            #Remove not alphanumeric symbols white spaces
            sentimentText = re.sub(r'[^\w]', ' ', sentimentText)

        return trainData

#stemming!!!
def tokenize(text):
        stemmer = SnowballStemmer("english")
        stems = [stemmer.stem(t) for t in word_tokenize(text)]
        return stems
        
#preproceccing data DONE
def getTrainFeature():
        trainData, test = readTrainData()
        #eliminate the user names since the stop words will be eliminated by scikit tfidfrepresentation
        trainData = eliminateStopWords(trainData)

    #     #this does the tokenizing as well and eliminates the stop words it also ignores puntuations
    #     #so basically we will get the vocabulary of every word used except the stop words
    #     #this is a dictionary of word:id pair
        vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False,
                                    sublinear_tf=True, stop_words=sstop_words, tokenizer=tokenize,
                                    analyzer='word')
    #     #this will give us the tf * idf weight for every word in our documents in (documentNR, wordID) -> tf_idf pairs
        train_features = vectorizer.fit_transform([sentimentText for (sentiment,sentimentText) in trainData])
        return trainData, train_features, vectorizer, test

#Naive Bayes.
def multinomial_naive_bayes(trainData, trainFeature):
        #Multinomial naive bayes
        nb = MultinomialNB()
        #setiment = 1 or 0 where 0 = negative, 1 = positive
        nb.fit(trainFeature, trainData)
        #trained naive bayes classifier that can do predictions
        return nb

#Nive Bayes DONE

#SVM
def svm_classifier(trainData, trainFeature, params):
        clf = SVC(C=params['C'], gamma=params['gamma'])
        # clf = svm.SVC()
        clf.fit(trainFeature, trainData)  
        return clf

def tune_svm_parameters(trainData, trainFeature):
        # parameters gamma and C of the Radial Basis Function (RBF) kernel SVM.
        # Intuitively, the gamma parameter defines how far the influence of a single training example
        #  reaches, with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters 
        #  can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.
        # The C parameter trades off misclassification of training examples against simplicity of the decision surface. 
        # A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly 
        # by giving the model freedom to select more samples as support vectors.

        C_range = numpy.logspace(-3, 10, 13)
        gamma_range = numpy.logspace(-9, 3, 12)
        param_grid = dict(gamma=gamma_range, C=C_range)

        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, return_train_score=True, verbose=10)
        # grid = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=cv, return_train_score=True)
        grid.fit(trainFeature, trainData)
        return grid.best_params_, grid.best_score_


def visualize_parameters(train_features, trainData,params):
    X = train_features.todense()
    pca = PCA(n_components=2).fit(X)
    data2D = pca.transform(X)
    h = .02  # step size in the mesh
    x_min, x_max = data2D[:, 0].min() - .5, data2D[:, 0].max() + .5
    y_min, y_max = data2D[:, 1].min() - .5, data2D[:, 1].max() + .5
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                         numpy.arange(y_min, y_max, h))
    clf = SVC(kernel='rbf',C=params['C'], gamma=params['gamma'])
    clf.fit(data2D, trainData)

    plt.figure(figsize=(1, 1))
    color_map = {'0': (0, 0, 1), '1': (1, 0, 0)}
    colors = [color_map[y] for y in trainData]
    Z = clf.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.title("gamma=10^%d, C=10^%d" % (numpy.log10(params['gamma']), numpy.log10(params['C'])),
              size='medium')
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(data2D[:, 0], data2D[:, 1], c=colors, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

    plt.show()

	
def plot_test2D(train_features, target):
    X = train_features.todense()
    pca = PCA(n_components=2).fit(X)
    data2D = pca.transform(X)
    color_map = {'0': (0, 0, 1), '1': (1, 0, 0)}
    colors = [color_map[y] for y in target]
    plt.scatter(data2D[:,0], data2D[:,1], c= colors)
    plt.show() 

def plot_test3D(train_features, target):
    X = train_features.todense()
    pca = PCA(n_components=3).fit(X)
    data3D = pca.transform(X)
    color_map = {'0': (0, 0, 1), '1': (1, 0, 0)}
    colors = [color_map[y] for y in target]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(data3D[:,0], data3D[:,1],data3D[:,2], c = colors)
    plt.show()

classify()
