import datetime
import os
from django.shortcuts import render, redirect
from . import apiai_controller as ai_controller
from . import models as model
from .utils import file_control as fc
from . import sentiment_analyzer
import pickle


# Create your views here.
def index(request):
    """ index """
    my_ai = ai_controller.AiController()
    messages = []
    filename = "conversation.txt"
    contents =  fc.read_from(filename)
    #the last message is the one that was sent to the bot right now
    text = my_ai.get_text_from_response(my_ai.send_text_message("12345678", contents[-1]))
    fc.append_to(filename, text)

    sentiment = pickle.load(open('sentiment_classifier.pkl', 'rb'))
    data = sentiment.vectorizer.transform([contents[-1]])
    # prediction = sentiment.nb_cls.predict(data)
    prediction = sentiment.svm_cls.predict(data)
    
    if prediction[0] == 0:
        sentiment = " = negative"
    else:
        sentiment = " = positive"
    messages.append(model.Message(text, datetime.datetime.now()))
    messages.append(model.Message(contents[-1]+sentiment, datetime.datetime.now()))

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
    file = open(filename, "a+")
    file.write('\n'+sent_text)
    file.close()
    return redirect('index')