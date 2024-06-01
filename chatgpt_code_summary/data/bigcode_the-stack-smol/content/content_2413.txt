"""Model module for images."""
from django.db import models
from django.contrib.auth.models import User
from imager_profile.models import ImagerProfile


# Create your models here.
class ImageBaseClass(models.Model):
    """Base class for Photo and Album classes."""

    PRIVATE = 'PRVT'
    SHARED = 'SHRD'
    PUBLIC = 'PBLC'

    PUBLISHED = ((PRIVATE, 'private'),
                 (SHARED, 'shared'),
                 (PUBLIC, 'public'))
    title = models.CharField(max_length=180)
    description = models.TextField(max_length=500, blank=True, null=True)
    date_modified = models.DateField(auto_now=True)
    date_published = models.DateField(blank=True, null=True)
    published = models.CharField(choices=PUBLISHED, max_length=8)


    class Meta:
        """Meta."""

        abstract = True


class Photo(ImageBaseClass):
    """Photo model."""

    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='photo')
    image = models.ImageField(upload_to='images')
    date_uploaded = models.DateField(editable=False, auto_now_add=True)
    
    def __str__(self):
        """Print function displays username."""
        return self.title


class Album(ImageBaseClass):
    """Album model."""

    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='album')
    cover = models.ImageField(upload_to='images')
    date_created = models.DateField(editable=False, auto_now_add=True)
    photos = models.ManyToManyField(Photo, related_name='albums', blank=True)

    def __str__(self):
        """Print function displays username."""
        return self.title
