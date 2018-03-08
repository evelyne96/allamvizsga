from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

# Create your models here.
class Profile(models.Model):
    """User model"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=254)
    last_name = models.CharField(max_length=254)
    email = models.CharField(max_length=254, blank=False)
    email_confirmed = models.BooleanField(default=False)
    character_name = models.CharField(max_length=254)
    companion = models.CharField(max_length=254, default='images/female/Koko/talk.png')
    mood = models.IntegerField(default=0)
    story_type = models.CharField(max_length=254)

    def __str__(self):
        return self.first_name

# @receiver(post_save, sender=User)
# def update_user_profile(sender, instance, created, **kwargs):
#     print('udtade user profile')
#     if created:
#         Profile.objects.create(user=instance)
#     try:
#         instance.profile.save()
#     except:
#         print("This user does not have a profile.")
