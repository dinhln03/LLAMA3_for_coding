"""Django models utilities."""

#Django
from django.db import models

class CRideModel(models.Model):
    """
    CrideModel acts as an abastract base class from
    which every other model in the project will inherit.
    This class provides every table with de following
    attributes:
        + created (Datetime): store the datetime the object was created
        + modified (Datetime): store the last datetime the object was modified
    """

    created = models.DateTimeField(
        'created at',
        auto_now_add=True, # set the date auto when the model is created
        help_text='Date time on which the object was created.'
    )
    modified = models.DateTimeField(
        'modified at',
        auto_now=True, # set the date when the model is called
        help_text='Date time on which the object was last modified.'
    )

    class Meta:
        """Meta options."""

        abstract = True
        # Set class config when it is called
        get_latest_by = 'created'
        ordering = ['-created', '-modified']

