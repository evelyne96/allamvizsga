import datetime
import os
from django.shortcuts import render, redirect
from . import apiai_controller as ai_controller
from . import models as model

# Create your views here.
def index(request):
    """ index """
    my_ai = ai_controller.AiController()
    messages = []
    filename = "conversation.txt"
    if os.path.exists(filename):
        file = open(filename, "r")
    else:
        file = open(filename, "w")
    file = open(filename, "r")
    last_one = "Joke"
    for line in file:
        messages.append(model.Message(line, datetime.datetime.now()))
        last_one = line
    text = my_ai.get_text_from_response(my_ai.send_text_message("12345678", last_one))
    file.close()
    file = open(filename, "a+")
    file.write('\n'+text)
    file.close()
    messages.append(model.Message(text, datetime.datetime.now()))

    context = {
        'title': "Chatbot.",
        'messages': messages,
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
