from django import forms

from sme_uniforme_apps.proponentes.models import Anexo


class AnexoForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super(AnexoForm, self).__init__(*args, **kwargs)

        self.fields['tipo_documento'].required = True

    class Meta:
        model = Anexo
        fields = '__all__'