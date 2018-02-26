
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.sites.shortcuts import get_current_site
from django.shortcuts import render, redirect
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from django.contrib.auth import logout

from .forms import SignUpForm
from .models import Profile

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=True)
            data = form.cleaned_data
            first_name = data['first_name']
            last_name = data['last_name']
            email = data['email']
            profile = Profile(user=user, first_name=first_name, last_name=last_name,email=email,character_name = 'images/male/Gamers/5.png', story_type = 'chatbot')
            profile.save()

            return redirect('/bot')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form, 'background' : "images/wallpaper/default.jpg",})

def login_view(request):
    print("hey")
    if request.user.is_authenticated:
        logout(request)
    return render(request, 'login.html', {'background' : "images/wallpaper/default.jpg",})

def logout_view(request):
    print("logout")
    logout(request)
    return redirect('authenticate.views.login_view')



