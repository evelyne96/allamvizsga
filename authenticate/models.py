from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from bot.models import BotCharacter

# Create your models here.
class Profile(models.Model):
    """User model"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=254)
    last_name = models.CharField(max_length=254)
    email = models.CharField(max_length=254, blank=False)
    email_confirmed = models.BooleanField(default=False)

    def __str__(self):
        return self.first_name

class UserCharacter(models.Model):
    name = models.CharField(max_length=254)
    gender = models.CharField(max_length=254, default='female')
    image = models.CharField(max_length=255)

class UserSettings(models.Model):
    """UserSetting model"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    user_character = models.ForeignKey(UserCharacter, on_delete=models.DO_NOTHING, default=1)
    companion_character = models.ForeignKey(BotCharacter,  on_delete=models.DO_NOTHING, default=1)
    mood = models.IntegerField(default=0)
    story_type = models.CharField(max_length=254)

    def update_mood(self, prediction):
        if prediction == 0 and self.mood > (-7):
            self.mood -= 1
            self.save()
            #negative
        elif self.mood < 7:
            #positive
            self.mood += 1
            self.save()

    def __str__(self):
        return self.user

# @receiver(post_save, sender=User)
# def update_user_profile(sender, instance, created, **kwargs):
#     print('udtade user profile')
#     if created:
#         Profile.objects.create(user=instance)
#     try:
#         instance.profile.save()
#     except:
#         print("This user does not have a profile.")
