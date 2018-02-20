import datetime
import os
from django.shortcuts import render, redirect
from . import apiai_controller as ai_controller
from . import models as model
from .utils import file_control as fc
from . import sentiment_analyzer
import pickle
from sklearn.metrics import accuracy_score
from django.http import HttpResponse

# Create your views here.
def index(request):
    """ index """
    my_ai = ai_controller.AiController()
    messages = []
    filename = "conversation.txt"
    contents =  fc.read_from(filename)
    # sentiment_anal = sentiment_analyzer.SentimentAnalyzer()
    # pickle.dump(sentiment_anal, open('sentiment_classifier.pkl', 'wb'))


    #the last message is the one that was sent to the bot right now
    if len(contents) != 0:
        text = my_ai.get_text_from_response(my_ai.send_text_message("12345678", contents[-1]))
        sentiment = pickle.load(open('sentiment_classifier.pkl', 'rb'))
        data = sentiment.vectorizer.transform([contents[-1]])
        # prediction = sentiment.nb_cls.predict(data)
        prediction = sentiment.svm_cls.predict(data)
    
        if prediction[0] == 0:
            sentiment = " = negative"
        else:
            sentiment = " = positive"
        messages.append(model.Message(contents[-1]+sentiment, datetime.datetime.now()))
    else:
         text = my_ai.get_text_from_response(my_ai.send_text_message("12345678", 'Hi'))
    messages.append(model.Message(text, datetime.datetime.now()))
    fc.write_to(filename, "")

    context = {
        'title': "Chatbot.",
        'messages': messages,
        'background' : "images/wallpaper/happy/2.jpg",
        'character' : "images/female/Koko/talk2.png",
        'gamer' : "images/male/Gamers/5.png",
        'characterName' : "Koko"
    }
    return render(request, 'index.html', context)

def post_message(request):
    """Post new message"""
    sent_text = request.POST['text']
    filename = "conversation.txt"
    fc.write_to(filename, sent_text)
    return redirect('index')

def test_sentiment(request):
    """Test"""
    filename = os.path.join(os.getcwd(), 'bot', 'static', 'sentimentAnalysis', 'train.tsv')
    text, textSentiment = fc.read_tsv(filename)
    sentiment = pickle.load(open('sentiment_classifier.pkl', 'rb'))
    test_features = sentiment.vectorizer.transform(text)
    prediction = sentiment.svm_cls.predict(test_features)
    prediction2 = sentiment.nb_cls.predict(test_features)
    svmPrecision = accuracy_score(textSentiment, prediction)
    nbPrecision = accuracy_score(textSentiment, prediction2)
    return HttpResponse("SVM = " + str(svmPrecision) + "\n" + "NB = " + str(nbPrecision))