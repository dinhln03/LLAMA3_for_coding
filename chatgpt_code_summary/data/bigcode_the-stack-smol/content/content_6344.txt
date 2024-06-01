from django import forms
from .models import Reclamacao,Login,Comentario
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class CadastraReclamacaoForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super(CadastraReclamacaoForm,self).__init__(*args, **kwargs)
        self.fields['titulo'].required = True
        self.fields['bairro'].required = True
        self.fields['rua'].required = True
      
        self.fields['descricao'].required = True
        self.fields['foto'].required = False
        
    class Meta:
        model = Reclamacao
        fields = ('titulo','bairro','rua','descricao', 'foto',)
 

class LoginUsuarioForm(forms.ModelForm):
	class Meta:
		model = Login
		fields = ('username','password',)
		widgets = {
        'password': forms.PasswordInput(),
    }
class SignUpForm(UserCreationForm):
    cpf = forms.CharField(max_length=11, required=True)
    bairro = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(max_length=254, help_text='Required. Inform a valid email address.')

    class Meta:
        model = User
        fields = ('username', 'cpf', 'bairro', 'email', 'password1', 'password2', )

#class CadastraForum(forms.ModelForm):
#    class Meta:
#        model = Forum
#        fields = ('text',)
class RegistroDeComentarioForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super(RegistroDeComentarioForm,self).__init__(*args, **kwargs)
        self.fields['text1'].required = True
        
    class Meta:
        model = Comentario
        fields = ('text1',)