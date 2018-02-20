from django.urls import path
from . import views
from django.views.generic.base import RedirectView
from django.contrib.staticfiles.storage import staticfiles_storage

urlpatterns = [
     path('', views.index, name='index'),
     path('post/message', views.post_message, name='post_message'),
     path('testSentiment', views.test_sentiment, name='test_sentiment'),
     path('favicon.ico', RedirectView.as_view( url=staticfiles_storage.url('/favicon.ico') ), name="favicon" ),
]