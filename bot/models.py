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