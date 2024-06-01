# -*- coding: utf-8 -*-
import uuid
import pytz
from faker import Faker
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.gis.db.models import PointField
from django.contrib.postgres.indexes import BrinIndex
from django.contrib.postgres.fields import JSONField
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.template.defaultfilters import slugify
from django.urls import reverse
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
from django.utils.translation import ugettext_lazy as _
from django.utils import timezone

from oauthlib.common import generate_token
from oauth2_provider.models import (
    Application,
    AbstractApplication,
    AbstractAccessToken,
    AccessToken,
    RefreshToken
)

from foundation.constants import *


class ProblemDataSheetManager(models.Manager):
    def delete_all(self):
        items = ProblemDataSheet.objects.all()
        for item in items.all():
            item.delete()

    # def seed(self, user, product, length=25):
    #     results = []
    #     faker = Faker('en_CA')
    #     for i in range(0,length):
    #         farm = ProblemDataSheet.objects.create(
    #             name = faker.domain_word(),
    #             description = faker.sentence(nb_words=6, variable_nb_words=True, ext_word_list=None),
    #             user = user,
    #             product = product,
    #         )
    #         results.append(farm)
    #     return results


class ProblemDataSheet(models.Model):
    """
    Class model represents danger / destructive element to a production crop.

    Special thanks:
    (1) Preventing, Diagnosing, and Correcting Common Houseplant Problems via URL
        https://extension.psu.edu/preventing-diagnosing-and-correcting-common-houseplant-problems
    """

    '''
    Metadata
    '''
    class Meta:
        app_label = 'foundation'
        db_table = 'mika_problem_data_sheet'
        verbose_name = _('Problem Data Sheet')
        verbose_name_plural = _('Problem Data Sheets')
        default_permissions = ()
        permissions = (
            # ("can_get_opening_hours_specifications", "Can get opening hours specifications"),
            # ("can_get_opening_hours_specification", "Can get opening hours specifications"),
            # ("can_post_opening_hours_specification", "Can create opening hours specifications"),
            # ("can_put_opening_hours_specification", "Can update opening hours specifications"),
            # ("can_delete_opening_hours_specification", "Can delete opening hours specifications"),
        )

    '''
    Constants & Choices
    '''

    class TYPE_OF:
        PEST = 1
        DISEASE = 2
        ABIOTIC = 3
        NONE = 4

    TYPE_OF_CHOICES = (
        (TYPE_OF.PEST, _('Pest')),
        (TYPE_OF.DISEASE, _('Disease')),
        (TYPE_OF.ABIOTIC, _('Abiotic')),
        (TYPE_OF.NONE, _('None')),
    )

    '''
    Object Managers
    '''
    objects = ProblemDataSheetManager()

    '''
    Fields
    '''

    #
    # Internal Related Fields
    #

    slug = models.SlugField(
        _("Slug"),
        help_text=_('The unique slug used for this crop when accessing details page.'),
        max_length=127,
        blank=True,
        null=False,
        db_index=True,
        unique=True,
        editable=False,
    )
    text = models.CharField(
        _("Text"),
        max_length=127,
        help_text=_('The variety name of the crop.'),
        blank=True,
        null=True,
        db_index=True,
    )
    type_of = models.PositiveSmallIntegerField(
        _("Type of"),
        help_text=_('The type of production crop problem.'),
        blank=False,
        null=False,
        choices=TYPE_OF_CHOICES,
    )

    '''
    Methods
    '''

    def __str__(self):
        return str(self.slug)
