from django.db import models

class Rice(models.Model):
    state_name=models.CharField("state_name",max_length=255)
    distict_name=models.CharField("distict_name",max_length=255)
    crop_year=models.IntegerField("crop_year")
    season=models.CharField("season",max_length=255)
    crop=models.CharField("crop",max_length=255)
    temperature=models.FloatField("temperature")
    precipitation=models.FloatField("precipitation")
    humidity=models.IntegerField("humidity")
    area=models.IntegerField("area")
    production=models.FloatField("production")