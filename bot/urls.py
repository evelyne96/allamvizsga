from django.urls import path
from . import views
from django.views.generic.base import RedirectView
from django.contrib.staticfiles.storage import staticfiles_storage

urlpatterns = [
     path('bot/', views.index, name='bot'),
     path('sendMessage/', views.post_message, name='post_message'),
     path('settings/', views.settings, name='settings'),
     path('post_settings/', views.post_settings, name='post_settings'),
     path('favicon.ico', RedirectView.as_view( url=staticfiles_storage.url('/favicon.ico') ), name="favicon" ),
]