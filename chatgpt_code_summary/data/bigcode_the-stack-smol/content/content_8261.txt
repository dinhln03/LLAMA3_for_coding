from django.shortcuts import render, HttpResponseRedirect, get_object_or_404
from cartapp.models import Cart
from mainapp.models import Product
from django.contrib.auth.decorators import login_required



@login_required
def view(request):
    return render(request, 'cartapp/cart.html', context = {
        'cart': Cart.objects.filter(user=request.user)
    })

@login_required
def add(request, product_id):

    product = get_object_or_404(Product, pk=product_id)
    
    cart_items = Cart.objects.filter(user=request.user, product=product)

    if cart_items:
        cart = cart_items.first()
    else:
        cart = Cart(user=request.user, product=product)

    cart.quantity += 1
    cart.save()
    
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
    
@login_required    
def remove(request, cart_item_id):
    cart = get_object_or_404( Cart, pk=cart_item_id )
    cart.delete()
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))

@login_required
def edit(request, cart_item_id, quantity):
    quantity = quantity
    cart_item = Cart.objects.get(pk=cart_item_id)
        
    if quantity > 0: 
        cart_item.quantity = quantity
        cart_item.save()
    else:
        cart_item.delete()
        
    return render(request, 'cartapp/include/inc_cart_edit.html')