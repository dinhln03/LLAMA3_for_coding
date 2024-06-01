from django.db import models

# Create your models here.
class Message(models.Model):
    tab_name = models.TextField(max_length=50)
    text = models.TextField(max_length=300)