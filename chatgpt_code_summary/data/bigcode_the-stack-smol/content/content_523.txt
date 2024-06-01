from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from django.db.models.signals import post_save


# Create your models here.
class Profile(models.Model):
	user = models.OneToOneField(settings.AUTH_USER_MODEL)
	date_of_birth = models.DateField(blank=True, null=True)
	photo = models.ImageField(upload_to='users/%Y/%m/%d', blank=True)
	course_bookmark = models.CharField(max_length=100, default='the-strategy')
	module_bookmark = models.PositiveIntegerField(default=0)

	def __str__(self):
		return 'Profile for user {}'.format(self.user.username)


class Contact(models.Model):
	user_from = models.ForeignKey(User,	related_name='rel_from_set')
	user_to = models.ForeignKey(User, related_name='rel_to_set')
	created = models.DateTimeField(auto_now_add=True, db_index=True)

	class Meta:
		ordering = ('-created',)
	
	def __str__(self):
		return '{} follows {}'.format(self.user_from, self.user_to)


User.add_to_class('following', models.ManyToManyField('self',
													  through=Contact,
													  related_name='followers',
													  symmetrical=False))

# Signal to auto-create a profile when a User is created.
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

post_save.connect(create_user_profile, sender=User)



