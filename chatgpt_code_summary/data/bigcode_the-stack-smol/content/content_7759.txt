from django.contrib import admin
from .models import *

admin.site.register(Pokemon)
admin.site.register(PokemonEvolution)
admin.site.register(PokemonStat)
admin.site.register(Stat)
admin.site.register(EvolutionChain)
