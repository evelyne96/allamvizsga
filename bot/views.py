import os,pickle,datetime,json
from django.shortcuts import render, redirect
from . import apiai_controller as ai_controller, models as model, sentiment_analyzer
from .utils import file_control as fc
from sklearn.metrics import accuracy_score
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from myapp import settings
from authenticate.models import Profile, UserCharacter, UserSettings

@login_required(login_url='/login/')
def index(request):
    """ index """
    # sentiment_anal = sentiment_analyzer.SentimentAnalyzer()
    # pickle.dump(sentiment_anal, open('sentiment_classifier.pkl', 'wb'))

    #sentiment = pickle.load(open('sentiment_classifier.pkl', 'rb'))

    my_ai = ai_controller.AiController()
    text = my_ai.get_text_from_response(my_ai.send_text_message("12345678", 'Hi'))
    messages = []
    messages.append(model.Message(text, datetime.datetime.now()))
    us = UserSettings.objects.filter(user_id = request.user.id)[0]
    us.mood = 0
    us.save()
    context = {
        'title': "Visual Novel",
        'message': text,
        'character' : us.companion_character,
        'gamer' : us.user_character,
        'mood' : us.mood
    }
  
    return render(request, 'content.html', context)

@login_required(login_url='/login/')
def post_message(request):
    """Post new message"""
    import json
    data = json.loads(request.body); sent_text = data['sent_text']
    my_ai = ai_controller.AiController()
    current_profile = UserSettings.objects.filter(user_id = request.user.id)[0]
    if len(sent_text) != 0:
        answer = my_ai.get_text_from_response(my_ai.send_text_message(request.session.session_key, sent_text))
        sentiment = pickle.load(open('sentiment_classifier.pkl', 'rb'))
        data = sentiment.vectorizer.transform([answer])
        prediction = sentiment.svm_cls.predict(data)
        current_profile.update_mood(prediction)
    else:
         answer = my_ai.get_text_from_response(my_ai.send_text_message(request.session.session_key, 'Hi'))

    data = { 'answer' : answer, 'mood' : current_profile.mood}
    return HttpResponse(json.dumps(data), content_type="application/json")

def settings(request):
    current_profile = Profile.objects.filter(user_id = request.user.id)[0]
    current_settings = UserSettings.objects.filter(user_id = request.user.id)[0]
    fgamers = model.BotCharacter.objects.filter(gender='female')
    print(fgamers)
    current_character = model.UserCharacter.objects.filter(user_id = request.user.id)[0]
    context = { 'gamers' : fgamers,
                'username' : current_profile.first_name,
                'userCharacter' : current_character.name
              }
    return render(request, 'settings.html', context)