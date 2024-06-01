from svs.models import Customer
from django.db import models
from django.utils import timezone
from svs.models import Customer, Machine
from core.models import CoreUser
from markdownx.models import MarkdownxField

STATUSES = (
    ("pending_our", "Pending - Our Side"),
    ("pending_their", "Pending - Their Side"),
    ("timeout", "More Than a Week"),
    ("closed", "Closed 😁"),
)


class Issue(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    title = models.CharField(max_length=255)
    contact = models.CharField(max_length=255)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    user = models.ForeignKey(CoreUser, on_delete=models.CASCADE)
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE)
    description = MarkdownxField()
    status = models.CharField(
        max_length=20, choices=STATUSES, null=False, default="pending_ours"
    )

    def __str__(self) -> str:
        return self.title


class IssueEntry(models.Model):
    issue = models.ForeignKey(Issue, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    title = models.CharField(max_length=255)
    description = MarkdownxField()

    def __str__(self) -> str:
        return self.title

    class Meta:
        verbose_name_plural = "Entries"
