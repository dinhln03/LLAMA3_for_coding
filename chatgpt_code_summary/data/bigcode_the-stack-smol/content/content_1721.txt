from django.http import HttpResponseRedirect
from django.views.generic import ListView,CreateView,UpdateView,DetailView,View
from django.shortcuts import render, redirect
from ecom import forms, models

from django.utils.decorators import method_decorator


def admin_required(function):
    def wrap(request, *args, **kwargs):
        if not request.user.groups.filter(name='Administrador').exists():
            return redirect('')
        return function(request, *args, **kwargs)

    return wrap

class Agregar_paquete_view(CreateView):
    # specify the model for create view
    model = models.Paquete
    form_class = forms.PaqueteForm
    # specify the fields to be displayed
    template_name = 'ecom/paquetes/Agregar_paquete.html'  # templete for updating
    success_url = "/Ver-paquete"

    @method_decorator(admin_required)
    def dispatch(self, request, *args, **kwargs):
        return super(Agregar_paquete_view, self).dispatch(request, *args, **kwargs)


class paquete_view(View):

    @method_decorator(admin_required)
    def dispatch(self, request, *args, **kwargs):
        return super(paquete_view, self).dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        paquete = models.Paquete.objects.all()
        return render(request, 'ecom/paquetes/Ver_paquete.html',{"paquete": paquete})

class Actualizar_paquete(UpdateView):
    model = models.Paquete #model
    fields = "__all__" # fields / if you want to select all fields, use "__all__"
    template_name = 'ecom/paquetes/Actualizar_paquete.html' # templete for updating
    success_url = "/Ver-paquete"

    @method_decorator(admin_required)
    def dispatch(self, request, *args, **kwargs):
        return super(Actualizar_paquete, self).dispatch(request, *args, **kwargs)


def paquetes(request):
    if 'product_ids' in request.COOKIES:
        product_ids = request.COOKIES['product_ids']

        counter = product_ids.split('|')
        product_count_in_cart = len(set(counter))
    else:
        product_count_in_cart = 0
    paquetes = models.Paquete.objects.all()
    return  render(request, 'ecom/paquetes/paquete.html',{"paquetes":paquetes,"product_count_in_cart":product_count_in_cart   })