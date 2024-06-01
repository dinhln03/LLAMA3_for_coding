from django.db import models

# Create your models here.

class BaseView(models.Model):
	title = models.CharField(max_length=256)

	def __unicode__(self):
		return self.title

class port1View(models.Model):
	def __unicode__(self):
		return self.title

class port2View(models.Model):
	title = models.CharField(max_length=256)

	def __unicode__(self):
		return self.title


class port3View(models.Model):
	title = models.CharField(max_length=256)

	def __unicode__(self):
		return self.title

class port4View(models.Model):
	title = models.CharField(max_length=256)

	def __unicode__(self):
		return self.title

class port5View(models.Model):
	title = models.CharField(max_length=256)

	def __unicode__(self):
		return self.title

class port6View(models.Model):
	title = models.CharField(max_length=256)

	def __unicode__(self):
		return self.title

	