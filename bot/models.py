from django.db import models

# Create your models here.
class User(models.Model):
    """User model"""
    full_name = models.CharField(max_length=70)
    nick_name = models.CharField(max_length=50)

    def __str__(self):
        return self.full_name


class Message(models.Model):
    """Messages model"""
    def __init__(self, text, time):
        super(Message, self).__init__()
        self.text = text
        self.time = time

    def __str__(self):
        return self.text

class TrainData(models.Model):
    """Sentiment training data."""
    def __init__(self,itemID, sentiment, sentimentText):
        super(TrainData, self).__init__()
        self.itemID = itemID
        self.sentiment = sentiment
        self.sentimentText = sentimentText
        self.words = []


    def __str__(self):
        return self.sentimentText

    def create_new(self, list):
        isCreated = False
        model = None
        if len(list) > 2:
            isCreated = True
            model = TrainData(list[0], list[1], list[2])
        return isCreated, model

