from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.contenttypes.models import ContentType
from django.utils.translation import ugettext_lazy as _

from djangocms_versioning.constants import PUBLISHED, VERSION_STATES
from djangocms_versioning.versionables import _cms_extension
from polymorphic.utils import get_base_polymorphic_model
from rangefilter.filters import DateRangeFilter

from .helpers import get_rangefilter_expires_default


class SimpleListMultiselectFilter(admin.SimpleListFilter):

    def value_as_list(self):
        return self.value().split(',') if self.value() else []

    def _update_query(self, changelist, include=None, exclude=None):
        selected_list = self.value_as_list()
        if include and include not in selected_list:
            selected_list.append(include)
        if exclude and exclude in selected_list:
            selected_list.remove(exclude)
        if selected_list:
            compiled_selection = ','.join(selected_list)
            return changelist.get_query_string({self.parameter_name: compiled_selection})
        else:
            return changelist.get_query_string(remove=[self.parameter_name])


class ContentTypeFilter(SimpleListMultiselectFilter):
    title = _("Content Type")
    parameter_name = "content_type"
    template = 'djangocms_content_expiry/multiselect_filter.html'

    def lookups(self, request, model_admin):
        lookup_list = []
        for content_model in _cms_extension().versionables_by_content:
            # Only add references to the inherited concrete model i.e. not referenced polymorphic models
            if hasattr(content_model, "polymorphic_ctype"):
                content_model = get_base_polymorphic_model(content_model)
            # Create an entry
            content_type = ContentType.objects.get_for_model(content_model)
            lookup_list_entry = (content_type.pk, content_type)
            # Only add unique entries
            if lookup_list_entry not in lookup_list:
                lookup_list.append(lookup_list_entry)
        return lookup_list

    def queryset(self, request, queryset):
        content_types = self.value()
        if not content_types:
            return queryset

        return queryset.filter(version__content_type__in=content_types.split(','))

    def choices(self, changelist):
        yield {
            'selected': self.value() is None,
            'query_string': changelist.get_query_string(remove=[self.parameter_name]),
            'display': 'All',
            'initial': True,
        }
        for lookup, title in self.lookup_choices:
            yield {
                'selected': str(lookup) in self.value_as_list(),
                'query_string': changelist.get_query_string({self.parameter_name: lookup}),
                'include_query_string': self._update_query(changelist, include=str(lookup)),
                'exclude_query_string': self._update_query(changelist, exclude=str(lookup)),
                'display': title,
            }


class VersionStateFilter(SimpleListMultiselectFilter):
    title = _("Version State")
    parameter_name = "state"
    default_filter_value = PUBLISHED
    show_all_param_value = "_all_"
    template = 'djangocms_content_expiry/multiselect_filter.html'

    def _is_default(self, filter_value):
        if self.default_filter_value == filter_value and self.value() is None:
            return True
        return False

    def _get_all_query_string(self, changelist):
        """
        If there's a default value set the all parameter needs to be provided
        however, if a default is not set the all parameter is not required.
        """
        # Default setting in use
        if self.default_filter_value:
            return changelist.get_query_string(
                {self.parameter_name:  self.show_all_param_value}
            )
        # Default setting not in use
        return changelist.get_query_string(remove=[self.parameter_name])

    def _is_all_selected(self):
        state = self.value()
        # Default setting in use
        if self.default_filter_value and state == self.show_all_param_value:
            return True
        # Default setting not in use
        elif not self.default_filter_value and not state:
            return True
        return False

    def _update_query(self, changelist, include=None, exclude=None):
        selected_list = self.value_as_list()

        if self.show_all_param_value in selected_list:
            selected_list.remove(self.show_all_param_value)

        if include and include not in selected_list:
            selected_list.append(include)
        if exclude and exclude in selected_list:
            selected_list.remove(exclude)
        if selected_list:
            compiled_selection = ','.join(selected_list)
            return changelist.get_query_string({self.parameter_name: compiled_selection})
        else:
            return changelist.get_query_string(remove=[self.parameter_name])

    def lookups(self, request, model_admin):
        return VERSION_STATES

    def queryset(self, request, queryset):
        state = self.value()
        # Default setting in use
        if self.default_filter_value:
            if not state:
                return queryset.filter(version__state=self.default_filter_value)
            elif state != "_all_":
                return queryset.filter(version__state__in=state.split(','))
        # Default setting not in use
        elif not self.default_filter_value and state:
            return queryset.filter(version__state__in=state.split(','))
        return queryset

    def choices(self, changelist):
        yield {
            "selected": self._is_all_selected(),
            "query_string": self._get_all_query_string(changelist),
            "display": _("All"),
            'initial': True,
        }
        for lookup, title in self.lookup_choices:
            lookup_value = str(lookup)
            yield {
                "selected":  str(lookup) in self.value_as_list() or self._is_default(lookup_value),
                "query_string": changelist.get_query_string(
                    {self.parameter_name: lookup}
                ),
                'include_query_string': self._update_query(changelist, include=str(lookup_value)),
                'exclude_query_string': self._update_query(changelist, exclude=str(lookup_value)),
                "display": title,
            }


class AuthorFilter(admin.SimpleListFilter):
    """
    An author filter limited to those users who have added expiration dates
    """
    title = _("Version Author")
    parameter_name = "created_by"

    def lookups(self, request, model_admin):
        from django.utils.encoding import force_text
        User = get_user_model()
        options = []
        qs = model_admin.get_queryset(request)
        authors = qs.values_list('version__created_by', flat=True).distinct()
        users = User.objects.filter(pk__in=authors)

        for user in users:
            options.append(
                (force_text(user.pk), user.get_full_name() or user.get_username())
            )
        return options

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(created_by=self.value()).distinct()
        return queryset


class ContentExpiryDateRangeFilter(DateRangeFilter):
    def queryset(self, request, queryset):
        queryset = super().queryset(request, queryset)

        # By default the widget should default to show a default duration and not all content
        # expiry records
        if not any('expires__range' in seed for seed in request.GET):
            default_gte, default_lte = get_rangefilter_expires_default()
            queryset = queryset.filter(expires__range=(default_gte, default_lte))

        return queryset
