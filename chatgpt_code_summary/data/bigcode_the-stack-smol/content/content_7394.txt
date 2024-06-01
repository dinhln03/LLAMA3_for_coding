from django.db import models
from osc_bge.users import models as user_models


# Create your models here.
class TimeStampedModel(models.Model):

    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        abstract = True

#Agent Head Table
class AgencyHead(TimeStampedModel):

    PROGRAM_CHOICES = (
        ('secondary', 'Secondary'),
        ('college', 'College'),
        ('camp', 'Camp'),
    )

    name = models.CharField(max_length=80, null=True, blank=True)
    location = models.CharField(max_length=140, null=True, blank=True)
    number_branches = models.CharField(max_length=80, null=True, blank=True)
    capacity_students = models.CharField(max_length=255, null=True, blank=True)
    commission = models.CharField(max_length=140, null=True, blank=True)
    promotion = models.CharField(max_length=255, null=True, blank=True)
    others = models.CharField(max_length=255, null=True, blank=True)
    comment = models.TextField(null=True, blank=True)

    def __str__(self):
        return "{}".format(self.name)


class AgencyProgram(TimeStampedModel):

    head = models.ForeignKey(AgencyHead, on_delete=models.CASCADE, null=True)
    program = models.CharField(max_length=80, null=True, blank=True)


#Agent Branch Table
class Agency(TimeStampedModel):

    head = models.ForeignKey(AgencyHead, on_delete=models.CASCADE, null=True, related_name='agent_branch')
    name = models.CharField(max_length=140, null=True, blank=True)
    location = models.CharField(max_length=140, null=True, blank=True)
    capacity_students = models.CharField(max_length=255, null=True, blank=True)
    commission = models.CharField(max_length=140, null=True, blank=True)
    promotion = models.CharField(max_length=255, null=True, blank=True)
    others = models.CharField(max_length=255, null=True, blank=True)
    comment = models.TextField(null=True, blank=True)

    def __str__(self):
        return "{}".format(self.name)

class AgencyBranchProgram(TimeStampedModel):

    branch = models.ForeignKey(Agency, on_delete=models.CASCADE, null=True)
    program = models.CharField(max_length=80, null=True, blank=True)

def set_filename_format(now, instance, filename):

    return "{schoolname}-{microsecond}".format(
        agentname=instance.agency,
        microsecond=now.microsecond,
        )

def agent_directory_path(instance, filename):
    now = datetime.datetime.now()
    path = "agents/{agentname}/{filename}".format(
        agentname=instance.agency,
        filename=set_filename_format(now, instance, filename),
    )
    return path


class AgencyHeadContactInfo(TimeStampedModel):

    LEVEL_CHOICES = (
        ('s', 'S'),
        ('a', 'A'),
        ('b', 'B'),
        ('c', 'C'),
        ('d', 'D'),
    )

    agent = models.ForeignKey(AgencyHead, on_delete=models.CASCADE, null=True)
    name = models.CharField(max_length=80, null=True, blank=True)
    contracted_date = models.DateTimeField(auto_now=True, null=True)
    phone = models.CharField(max_length=80, null=True, blank=True)
    email = models.CharField(max_length=140, null=True, blank=True)
    skype = models.CharField(max_length=80, null=True, blank=True)
    wechat = models.CharField(max_length=80, null=True, blank=True)
    location = models.CharField(max_length=140, null=True, blank=True)
    level = models.CharField(max_length=80, null=True, blank=True)
    image = models.ImageField(upload_to=agent_directory_path, null=True, blank=True)

    def __str__(self):
        return "{}".format(self.name)


class AgentRelationshipHistory(TimeStampedModel):

    head = models.ForeignKey(AgencyHead, on_delete=models.CASCADE, null=True)
    writer = models.CharField(max_length=80, null=True, blank=True)
    name = models.CharField(max_length=80, null=True, blank=True)
    date = models.DateField(null=True, blank=True)
    location = models.CharField(max_length=140, null=True, blank=True)
    category = models.CharField(max_length=80, null=True, blank=True)
    priority = models.IntegerField(null=True, blank=True)
    comment = models.TextField(null=True, blank=True)


class SecodnaryProgram(TimeStampedModel):

    agent = models.ForeignKey(AgencyHead, on_delete=models.CASCADE, null=True)
    preriod = models.CharField(max_length=80, null=True, blank=True)
    target = models.IntegerField(null=True, blank=True)
    new_students_fall = models.IntegerField(null=True, blank=True)
    new_students_spring = models.IntegerField(null=True, blank=True)
    total_new_students_bge = models.IntegerField(null=True, blank=True)
    total_students_bge = models.IntegerField(null=True, blank=True)
    terminating_students = models.IntegerField(null=True, blank=True)
    comments = models.TextField(null=True, blank=True)


class Camp(TimeStampedModel):

    agent = models.ForeignKey(AgencyHead, on_delete=models.CASCADE, null=True)
    preriod = models.CharField(max_length=80, null=True, blank=True)
    target = models.IntegerField(null=True, blank=True)
    summer_camp = models.IntegerField(null=True, blank=True)
    winter_camp = models.IntegerField(null=True, blank=True)
    comments = models.TextField(null=True, blank=True)


class CollegeApplication(TimeStampedModel):

    agent = models.ForeignKey(AgencyHead, on_delete=models.CASCADE, null=True)
    preriod = models.CharField(max_length=80, null=True, blank=True)
    college_application = models.IntegerField(null=True, blank=True)
    other_program = models.IntegerField(null=True, blank=True)
    comments = models.TextField(null=True, blank=True)
