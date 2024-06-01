from django.utils import timezone
from maestros.models import Unidades
from maestros_generales.models import Empresas

__author__ = 'julian'

from django.contrib.gis.db import models
import datetime


class WaspTypeSensor(models.Model):
    name            = models.CharField(max_length=50)
    units           = models.ForeignKey(Unidades)
    fechaalta       = models.DateField(auto_now_add=True,verbose_name=("Fecha Alta"),blank=True,null=True)
    fechabaja       = models.DateField(verbose_name=("Fecha Baja"), blank=True,null=True)


class WaspMote(models.Model):
    DeviceName      = models.CharField(max_length=30)
    Imei            = models.BigIntegerField()
    fechaalta       = models.DateField(auto_now_add=True,verbose_name=("Fecha Alta"),blank=True,null=True)
    fechabaja       = models.DateField(verbose_name=("Fecha Baja"), blank=True,null=True)
    empresa         = models.ForeignKey(Empresas,null=True, blank=True,verbose_name=('Empresa'),on_delete=models.PROTECT)


class WaspSensor(models.Model):
    waspmote        = models.ForeignKey(WaspMote, on_delete=models.PROTECT)
    probestype      = models.ForeignKey(WaspTypeSensor,on_delete=models.PROTECT)
    fechaalta       = models.DateField(auto_now_add=True,verbose_name=("Fecha Alta"),blank=True,null=True)
    fechabaja       = models.DateField(verbose_name=("Fecha Baja"), blank=True,null=True)
    empresa         = models.ForeignKey(Empresas,null=True, blank=True,verbose_name=('Empresa'),on_delete=models.PROTECT)

class WaspData(models.Model):
    waspsensor         = models.ForeignKey(WaspSensor)
    timestamp_waspmote = models.DateTimeField()
    status             = models.CharField(max_length=1)
    #loc                = models.PointField(srid=4326)
    alt                = models.FloatField()
    lat                = models.FloatField()
    long               = models.FloatField()
    speed              = models.FloatField()
    course             = models.FloatField()
    voltage            = models.IntegerField()
    notes              = models.TextField()
    #objects            = models.GeoManager()
    valorsensor        = models.FloatField()
    #timestamp_server   = models.DateTimeField()
    timestamp_server   = models.DateTimeField(default= lambda: timezone.now() + datetime.timedelta(hours=1), blank=True)


