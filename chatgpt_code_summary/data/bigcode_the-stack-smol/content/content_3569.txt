import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils import timezone
from django.utils.translation import ugettext as _

from rr.forms.contact import ContactForm
from rr.models.contact import Contact
from rr.utils.serviceprovider import get_service_provider

logger = logging.getLogger(__name__)


@login_required
def contact_list(request, pk):
    """
    Displays a list of :model:`rr.Contact` linked to
    :model:`rr.ServiceProvider`.

    Includes a ModelForm for adding :model:`rr.Contact` to
    :model:`rr.ServiceProvider`.

    **Context**

    ``object_list``
        List of :model:`rr.Contact`.

    ``form``
        ModelForm for creating a :model:`rr.Contact`

    ``object``
        An instance of :model:`rr.ServiceProvider`.

    **Template:**

    :template:`rr/contact.html`
    """
    sp = get_service_provider(pk, request.user)
    form = ContactForm(sp=sp)
    if request.method == "POST":
        if "add_contact" in request.POST:
            form = _add_contact(request, sp)
        elif "remove_contact" in request.POST:
            _remove_contacts(request, sp)
    contacts = Contact.objects.filter(sp=sp, end_at=None)
    return render(request, "rr/contact.html", {'object_list': contacts,
                                               'form': form,
                                               'object': sp})


def _add_contact(request, sp):
    form = ContactForm(request.POST, sp=sp)
    if form.is_valid():
        contact_type = form.cleaned_data['type']
        firstname = form.cleaned_data['firstname']
        lastname = form.cleaned_data['lastname']
        email = form.cleaned_data['email']
        Contact.objects.create(sp=sp,
                               type=contact_type,
                               firstname=firstname,
                               lastname=lastname,
                               email=email)
        sp.save_modified()
        logger.info("Contact added for {sp} by {user}"
                    .format(sp=sp, user=request.user))
        messages.add_message(request, messages.INFO, _('Contact added.'))
        form = ContactForm(sp=sp)
    return form


def _remove_contacts(request, sp):
    for key, value in request.POST.dict().items():
        if value == "on":
            contact = Contact.objects.get(pk=key)
            if contact.sp == sp:
                contact.end_at = timezone.now()
                contact.save()
                sp.save_modified()
                logger.info("Contact removed from {sp} by {user}"
                            .format(sp=sp, user=request.user))
                messages.add_message(request, messages.INFO, _('Contact removed.'))