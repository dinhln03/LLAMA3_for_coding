from django.db import models
from .validators import validate_resume_ext

class Resume(models.Model):
    name = models.CharField(max_length = 20)
    phone = models.IntegerField()
    email = models.EmailField()
    resume = models.FileField(upload_to='resume/%Y/%m/%d/', validators=[validate_resume_ext])
    uploaded_at = models.DateTimeField(auto_now_add=True)

#Add name, phone number and email fields