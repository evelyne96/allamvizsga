from django.urls import path
from . import views
from django.views.generic.base import RedirectView
from django.contrib.staticfiles.storage import staticfiles_storage

urlpatterns = [
     path('bot/', views.index, name='bot'),
     path('post/message', views.post_message, name='post_message'),
     path('settings/', views.settings, name='settings'),
     path('testSentiment', views.test_sentiment, name='test_sentiment'),
     path('favicon.ico', RedirectView.as_view( url=staticfiles_storage.url('/favicon.ico') ), name="favicon" ),
]