from django.db import models

class BotCharacter(models.Model):
    name = models.CharField(max_length=254)
    gender = models.CharField(max_length=254, default='female')
    image = models.CharField(max_length=255)

class Message(models.Model):
    """Messages model"""
    text = models.CharField(max_length=70)
    time = models.DateField()

    def __str__(self):
        return self.text

class TrainData(models.Model):
    """Sentiment training data."""
    def __init__(self,itemID, sentiment, sentimentText):
        super(TrainData, self).__init__()
        self.itemID = itemID
        self.sentiment = sentiment
        self.sentimentText = sentimentText


    def __str__(self):
        return self.sentimentText

    def create_new(self, list):
        isCreated = False
        model = None
        if len(list) > 2:
            isCreated = True
            try:
                model = TrainData(list[0], int(list[1]), list[2])
            except ValueError:
                model = TrainData(list[0], 0, list[2])
        #this is for the test case
        if len(list) == 2:
            isCreated = True
            model = TrainData(list[0], 0, list[1])
        return isCreated, model

