from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User

class SignUpForm(UserCreationForm):
    email = forms.EmailField(max_length=254, help_text='Required. Inform a valid email address.')
    first_name = forms.CharField(max_length=70, help_text='First Name')
    last_name = forms.CharField(max_length=70, help_text='Last Name')

    
    def __init__(self, *args, **kwargs):
        super(UserCreationForm, self).__init__(*args, **kwargs)
        self.fields['password1'].help_text = 'PASSWORD'
        self.fields['password2'].help_text = 'VERIFY PASSWORD'
        self.fields['username'].help_text = 'USERNAME'
        self.fields['email'].help_text = 'asdffg@soemthing.com'
        self.fields['first_name'].help_text = 'FIRST NAME'
        self.fields['last_name'].help_text = 'LAST NAME'
        for fieldname in['username', 'password1', 'password2','email', 'last_name', 'first_name']:
            self.fields[fieldname].widget.attrs.update({'placeholder' : self.fields[fieldname].help_text})
            self.fields[fieldname].help_text = None
    
    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password1', 'password2', )


class LoginForm(AuthenticationForm):

     def __init__(self, *args, **kwargs):
        self.request = kwargs.pop('request', None)
        super(AuthenticationForm, self).__init__(*args, **kwargs)
        print("Loginform")
        self.fields['username'].widget.attrs.update({'placeholder' : 'USERNAME'})
        self.fields['password'].widget.attrs.update({'placeholder' : 'PASSWORD'})