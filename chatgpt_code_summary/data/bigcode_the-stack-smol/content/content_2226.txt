from __future__ import unicode_literals
from django.db import models
from model_utils import Choices
from model_utils.models import TimeStampedModel, StatusModel
from django.conf import settings


class Booking(TimeStampedModel, StatusModel):
    TRAVEL_CLASSES = Choices('economy', 'first_class')
    STATUS = Choices(('pending', 'Pending'), ('paid', 'Paid'), ('failed', 'Failed'))
    date_of_travel = models.DateTimeField()
    travel_class = models.CharField(choices=TRAVEL_CLASSES, default=TRAVEL_CLASSES.economy, max_length=30)
    status = models.CharField(choices=STATUS, default=STATUS.pending, max_length=20)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=False, blank=False)
    payment_reference = models.CharField(max_length=100, blank=True)
