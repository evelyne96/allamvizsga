from django.urls import path
from . import views

urlpatterns = [
     path('', views.index, name='index'),
    path('post/message', views.post_message, name='post_message')
]