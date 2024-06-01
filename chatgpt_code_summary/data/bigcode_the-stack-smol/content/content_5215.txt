from django.contrib import messages, auth
from django.contrib.auth.decorators import login_required
from payments.forms import MakePaymentForm
from django.shortcuts import render, get_object_or_404, redirect
from django.core.urlresolvers import reverse
from django.template.context_processors import csrf
from django.conf import settings
from services.models import Service


import stripe



stripe.api_key = settings.STRIPE_SECRET


@login_required(login_url="/accounts/login?next=payments/buy_now")
def buy_now(request, id):
    if request.method == 'POST':
        form = MakePaymentForm(request.POST)
        if form.is_valid():
            try:
                # service = get_object_or_404(Service, pk=id)
                customer = stripe.Charge.create(
                    amount= int(total*100),
                    currency="EUR",
                    description=request.user.email,
                    card=form.cleaned_data['stripe_id'],
                )
            except stripe.error.CardError:
                messages.error(request, "Your card was declined!")

            if customer.paid:
                messages.success(request, "You have successfully paid")
                return redirect(reverse('all_services'))
            else:
                messages.error(request, "Unable to take payment")
        else:
            messages.error(request, "We were unable to take a payment with that card!")

    else:
        form = MakePaymentForm()
        services = get_object_or_404(Service, pk=id)

    args = {'form': form, 'publishable': settings.STRIPE_PUBLISHABLE, 'services': services}
    args.update(csrf(request))

    return render(request, 'pay.html', args)