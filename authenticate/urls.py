from django.urls import path
from . import views
from . import forms
from django.views.generic.base import RedirectView
from django.contrib.staticfiles.storage import staticfiles_storage
from django.contrib.auth import views as auth_views

urlpatterns = [
     path('signup/', views.signup, name='signup'),
     path('login/', auth_views.login, {'template_name': 'login.html', 'authentication_form': forms.LoginForm}, name="login"),
     path('logout/', auth_views.logout, {'next_page': '/login'}, name="logout"),
     path('favicon.ico', RedirectView.as_view( url=staticfiles_storage.url('/favicon.ico') ), name="favicon" ),
]