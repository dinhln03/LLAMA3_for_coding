from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
from django.utils import timezone
# Create your models here.
class Article(models.Model):
        title=models.CharField(max_length=100)
        slug=models.SlugField(blank=True)
        body= models.TextField()
        date= models.DateTimeField(default=timezone.now)
        thumb=models.ImageField(default='default.jpg',blank=True)
        Author= models.ForeignKey(User,default=None,on_delete=models.CASCADE)
        
#Thumbnails

        def __str__(self):
           return self.title

        def snippets(self):
           return  self.body[:80] + '...'
