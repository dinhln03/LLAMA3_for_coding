from rest_framework import serializers
from vianeyRest.models import Usuario,Materia,Persona

class UsuarioSerializer(serializers.ModelSerializer):

    class Meta:
        model = Usuario
        fields = ('id','nombreUsuario','contrasenaUsuario')


class MateriaSerializer(serializers.ModelSerializer):

    class Meta:
        model = Materia
        fields = ('id','nombreMateria','primerParcialMateria',
                  'segundoParcialMateria','tercerParcialMateria',
                  'ordinarioMateria')


class PersonaSerializer(serializers.ModelSerializer):

    class Meta:
        model = Persona
        fields = ('id','nombrePersona','apeliidoPPersona',
                  'apellidoMPersona','licenciaturaPersona',
                  'semestrePersona')