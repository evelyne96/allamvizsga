import os,pickle,datetime,json
from django.shortcuts import render, redirect
from . import apiai_controller as ai_controller, models as model, sentiment_analyzer
from .utils import file_control as fc
from sklearn.metrics import accuracy_score
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from myapp import settings
from authenticate.models import Profile 

@login_required(login_url='/login/')
def index(request):
    """ index """
    # sentiment_anal = sentiment_analyzer.SentimentAnalyzer()
    # pickle.dump(sentiment_anal, open('sentiment_classifier.pkl', 'wb'))

    #sentiment = pickle.load(open('sentiment_classifier.pkl', 'rb'))

    my_ai = ai_controller.AiController()
    messages = []
    text = my_ai.get_text_from_response(my_ai.send_text_message("12345678", 'Hi'))
    messages.append(model.Message(text, datetime.datetime.now()))
    current_profile = getProfileByUserId(request.user.id)
    current_profile.mood = 0
    current_profile.save()
    context = {
        'title': "Visual Novel",
        'message': text,
        'character' : current_profile.companion,
        'gamer' : current_profile.character_name,
        'mood' : current_profile.mood
    }
    return render(request, 'content.html', context)

@login_required(login_url='/login/')
def post_message(request):
    """Post new message"""
    import json
    data = json.loads(request.body); sent_text = data['sent_text']
    my_ai = ai_controller.AiController()
    current_profile = getProfileByUserId(request.user.id)
    if len(sent_text) != 0:
        answer = my_ai.get_text_from_response(my_ai.send_text_message(request.session.session_key, sent_text))
        sentiment = pickle.load(open('sentiment_classifier.pkl', 'rb'))
        data = sentiment.vectorizer.transform([answer])
        # prediction = sentiment.nb_cls.predict(data)
        prediction = sentiment.svm_cls.predict(data)
        if prediction[0] == 0 and current_profile.mood > (-7):
            current_profile.mood -= 1
            current_profile.save()
            #negative
        elif current_profile.mood < 7:
            #positive
            current_profile.mood += 1
            current_profile.save()
        # m = model.Message(text = sent_text+sentiment,time=datetime.datetime.now())
        # m.save()
    else:
         answer = my_ai.get_text_from_response(my_ai.send_text_message(request.session.session_key, 'Hi'))

    data = { 'answer' : answer, 'mood' : current_profile.mood}
    return HttpResponse(json.dumps(data), content_type="application/json")

def settings(request):
    current_profile = getProfileByUserId(request.user.id)
    gamers = getGamersName()
    print(current_profile.character_name)
    context = { 'gamers' : gamers,
                'username' : current_profile.first_name,
                'userCharacter' : current_profile.character_name
              }
    return render(request, 'settings.html', context)


def getProfileByUserId(id):
    current_profile = Profile.objects.filter(user_id=id)
    return current_profile[0]

def getGamersName():
    n = 1
    gamers = []
    while n < 6 :
        male = "images/male/Gamers/"+str(n)+".png"
        female = "images/female/Gamers/"+str(n)+".png"
       # gamers.append(male)
        gamers.append(female)
        n += 1
    return gamers

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