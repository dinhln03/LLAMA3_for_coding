from django.db import models
from django.conf import settings

from django_extensions.db.models import TimeStampedModel

from djcommerce.utils import get_address_model

Address = get_address_model()

class Profile(TimeStampedModel):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete = models.CASCADE
    )
    addresses = models.ManyToManyField(Address)

    class Meta:
        abstract = False
        if hasattr(settings,"PROFILE_MODEL"):
            abstract = True
