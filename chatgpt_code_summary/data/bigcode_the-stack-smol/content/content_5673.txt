from adminsortable2.admin import SortableAdminMixin
from decimal import Decimal
from django.contrib import admin
from django.contrib.gis import admin as geo_admin
from import_export import fields
from import_export import widgets
from import_export.admin import ImportExportModelAdmin
from import_export.resources import ModelResource as ImportExportModelResource
from solo.admin import SingletonModelAdmin
from .models import Client
from .models import Contact
from .models import Contract
from .models import DashboardItem
from .models import Estimate
from .models import Invoice
from .models import Location
from .models import Log
from .models import Newsletter
from .models import Note
from .models import Profile
from .models import Project
from .models import Proposal
from .models import Report
from .models import Service
from .models import SettingsApp
from .models import SettingsCompany
from .models import SettingsContract
from .models import Task
from .models import Testimonial
from .models import Time


class BooleanWidget(widgets.Widget):
    """
    Convert strings to boolean values
    """

    def clean(self, value):
        if value == 'Yes':
            return True
        else:
            return False


class DecimalWidget(widgets.Widget):
    """
    Convert strings to decimal values
    """

    def clean(self, value):
        if value:
            return Decimal(value.replace(',', ''))
        else:
            return Decimal(0)


class UserWidget(widgets.Widget):
    """
    """

    def clean(self, value):
        return value


# Register your models here.


class ClientResource(ImportExportModelResource):
    """
    """

    class Meta:
        model = Client

    # auto fill id? #295
    # https://github.com/django-import-export/django-import-export/issues/295
    def get_instance(self, instance_loaders, row):
        return False

    def before_import(self, dataset, dry_run, file_name=None, user=None):

        if dataset.headers:
            dataset.headers = [
                str(header).lower().strip() for header in dataset.headers
            ]

        if 'id' not in dataset.headers:
            dataset.headers.append('id')


@admin.register(Client)
class ClientAdmin(ImportExportModelAdmin):
    """
    """
    resource_class = ClientResource


class ContactResource(ImportExportModelResource):
    """
    """
    client = fields.Field(
        column_name='client',
        attribute='client',
        widget=widgets.ForeignKeyWidget(Client, 'name'))

    class Meta:
        model = Contact

    def get_instance(self, instance_loaders, row):
        return False

    def before_import(self, dataset, dry_run, file_name=None, user=None):

        if dataset.headers:
            dataset.headers = [
                str(header).lower().strip() for header in dataset.headers
            ]

        if 'id' not in dataset.headers:
            dataset.headers.append('id')


@admin.register(Contact)
class ContactAdmin(ImportExportModelAdmin):
    """
    """
    resource_class = ContactResource


@admin.register(Contract)
class ContractAdmin(ImportExportModelAdmin):
    """
    """


@admin.register(DashboardItem)
class DashboardItemAdmin(SortableAdminMixin, admin.ModelAdmin):
    """
    """


class EstimateResource(ImportExportModelResource):
    """
    """
    client = fields.Field(
        column_name='client',
        attribute='client',
        widget=widgets.ForeignKeyWidget(Client, 'name'))
    amount = fields.Field(
        column_name='estimate_amount',
        attribute='amount',
        widget=DecimalWidget())
    subtotal = fields.Field(
        column_name='subtotal', attribute='subtotal', widget=DecimalWidget())
    document_id = fields.Field(
        column_name='estimate_id',
        attribute='document_id',
        widget=DecimalWidget())

    class Meta:
        model = Estimate

    def get_instance(self, instance_loaders, row):
        return False

    def before_import(self, dataset, dry_run, file_name=None, user=None):

        if dataset.headers:
            dataset.headers = [
                str(header).lower().strip() for header in dataset.headers
            ]

        if 'id' not in dataset.headers:
            dataset.headers.append('id')


@admin.register(Estimate)
class EstimateAdmin(ImportExportModelAdmin):
    """
    """
    resource_class = EstimateResource


class InvoiceResource(ImportExportModelResource):
    """
    """

    client = fields.Field(
        column_name='client',
        attribute='client',
        widget=widgets.ForeignKeyWidget(Client, 'name'))
    amount = fields.Field(
        column_name='amount', attribute='amount', widget=DecimalWidget())
    paid_amount = fields.Field(
        column_name='paid_amount',
        attribute='paid_amount',
        widget=DecimalWidget())
    subtotal = fields.Field(
        column_name='subtotal', attribute='subtotal', widget=DecimalWidget())
    balance = fields.Field(
        column_name='balance', attribute='balance', widget=DecimalWidget())
    document_id = fields.Field(
        column_name='invoice_id',
        attribute='document_id',
        widget=DecimalWidget())

    class Meta:
        model = Invoice

    def get_instance(self, instance_loaders, row):
        return False

    def before_import(self, dataset, dry_run, file_name=None, user=None):

        if dataset.headers:
            dataset.headers = [
                str(header).lower().strip() for header in dataset.headers
            ]

        if 'id' not in dataset.headers:
            dataset.headers.append('id')


@admin.register(Invoice)
class InvoiceAdmin(ImportExportModelAdmin):
    """
    """
    resource_class = InvoiceResource


@admin.register(Location)
class LocationAdmin(geo_admin.OSMGeoAdmin):
    """
    """
    search_fields = ('name', )


@admin.register(Log)
class LogAdmin(ImportExportModelAdmin):
    """
    """


@admin.register(Newsletter)
class NewsletterAdmin(ImportExportModelAdmin):
    """
    """


@admin.register(Note)
class NoteAdmin(ImportExportModelAdmin):
    """
    """


class ProjectResource(ImportExportModelResource):
    """
    """
    client = fields.Field(
        column_name='client',
        attribute='client',
        widget=widgets.ForeignKeyWidget(Client, 'name'))
    billable_amount = fields.Field(
        column_name='billable_amount',
        attribute='billable_amount',
        widget=DecimalWidget())
    budget = fields.Field(
        column_name='budget', attribute='budget', widget=DecimalWidget())
    budget_spent = fields.Field(
        column_name='budget_spent',
        attribute='budget_spent',
        widget=DecimalWidget())
    team_costs = fields.Field(
        column_name='team_costs',
        attribute='team_costs',
        widget=DecimalWidget())
    total_costs = fields.Field(
        column_name='total_costs',
        attribute='total_costs',
        widget=DecimalWidget())

    class Meta:
        model = Project
        exclude = ('task', 'team')

    def get_instance(self, instance_loaders, row):
        return False

    def before_import(self, dataset, dry_run, file_name=None, user=None):

        if dataset.headers:
            dataset.headers = [
                str(header).lower().strip() for header in dataset.headers
            ]

        if 'id' not in dataset.headers:
            dataset.headers.append('id')


@admin.register(Profile)
class ProfileAdmin(ImportExportModelAdmin):
    """
    """


@admin.register(Project)
class ProjectAdmin(ImportExportModelAdmin):
    """
    """
    resource_class = ProjectResource


@admin.register(Proposal)
class ProposalAdmin(ImportExportModelAdmin):
    """
    """


@admin.register(Report)
class ReportAdmin(ImportExportModelAdmin):
    """
    """


@admin.register(Service)
class ServiceAdmin(ImportExportModelAdmin):
    """
    """


@admin.register(SettingsApp)
class SettingsAppAdmin(SingletonModelAdmin):
    """
    """


@admin.register(SettingsCompany)
class SettingsCompanyAdmin(SingletonModelAdmin):
    """
    """


@admin.register(SettingsContract)
class SettingsContractAdmin(SingletonModelAdmin):
    """
    """


@admin.register(Testimonial)
class TestimonialAdmin(ImportExportModelAdmin):
    """
    """
    prepopulated_fields = {"slug": ("name", )}


class TaskResource(ImportExportModelResource):
    """
    """

    class Meta:
        model = Task
        exclude = ('unit', 'billable', 'active')

    def get_instance(self, instance_loaders, row):
        return False

    def before_import(self, dataset, dry_run, file_name=None, user=None):

        if dataset.headers:
            dataset.headers = [
                str(header).lower().strip() for header in dataset.headers
            ]

        if 'id' not in dataset.headers:
            dataset.headers.append('id')


@admin.register(Task)
class TaskAdmin(ImportExportModelAdmin):
    """
    """
    resource_class = TaskResource


class TimeResource(ImportExportModelResource):
    """
    """
    billable = fields.Field(
        column_name='billable', attribute='billable', widget=BooleanWidget())
    client = fields.Field(
        column_name='client',
        attribute='client',
        widget=widgets.ForeignKeyWidget(Client, 'name'))
    invoiced = fields.Field(
        column_name='invoiced', attribute='invoiced', widget=BooleanWidget())
    project = fields.Field(
        column_name='project',
        attribute='project',
        widget=widgets.ForeignKeyWidget(Project, 'name'))
    task = fields.Field(
        column_name='task',
        attribute='task',
        widget=widgets.ForeignKeyWidget(Task, 'name'))
    user = fields.Field(
        column_name='user', attribute='user', widget=UserWidget())

    class Meta:
        model = Time

    def get_instance(self, instance_loaders, row):
        return False

    def before_import(self, dataset, dry_run, file_name=None, user=None):

        if dataset.headers:
            dataset.headers = [
                str(header).lower().strip() for header in dataset.headers
            ]

        if 'id' not in dataset.headers:
            dataset.headers.append('id')


@admin.register(Time)
class TimeAdmin(ImportExportModelAdmin):
    """
    """
    resource_class = TimeResource
