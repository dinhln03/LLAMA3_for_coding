from django.views.generic import (
    ListView,
    DetailView,
    CreateView,
    UpdateView,
    DeleteView,
)
from django.urls import reverse_lazy
from .models import Pokemon

class PokemonListView(ListView):
    template_name = "pages/pokemon_list.html"
    model = Pokemon

class PokemonDetailView(DetailView):
    template_name = "pages/pokemon_detail.html"
    model = Pokemon

class PokemonCreateView(CreateView):
    template_name = "pages/pokemon_create.html"
    model = Pokemon
    fields = ['name', 'description', 'owner']

class PokemonUpdateView(UpdateView):
    template_name = "pages/pokemon_update.html"
    model = Pokemon
    fields = ['name', 'description', 'owner']

class PokemonDeleteView(DeleteView):
    template_name = "pages/pokemon_delete.html"
    model = Pokemon
    success_url = reverse_lazy("pokemon_list")
