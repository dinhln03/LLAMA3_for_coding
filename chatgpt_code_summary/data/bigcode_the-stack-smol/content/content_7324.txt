from django.utils import timezone
from .forms import SchedDayForm

class AdminCommonMixin(object):
    """
    common methods for all admin class
    set default values for owner, date, etc
    """

    def save_model(self, request, obj, form, change):
        try:
            obj.created_by = request.user
        except:
            pass
        super().save_model(request, obj, form, change)

    def get_queryset(self, request):
        """
        read queryset if is superuser
        or read owns objects
        """
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        return qs.filter(created_by=request.user)

    def response_change(self, request, obj):
        """
        get from response change some custom action from post
        ej: '_custom_action' in request.POST:
        """
        if '_custom_action' in request.POST:
            pass
        return super().response_change(request, obj)

    def response_add(self, request, obj):
        """
        get from response change some custom action from post
        ej: '_custom_action' in request.POST:
        """
        if '_custom_action' in request.POST:
            pass
        return super().response_add(request, obj)


class CalendarActionMixin(object):

    def save_model(self, request, obj, form, change):
        try:
            obj.created_by = request.user
        except:
            pass
        super().save_model(request, obj, form, change)

    def get_queryset(self, request):
        """
        read queryset if is superuser
        or read owns objects
        """
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        return qs.filter(created_by=request.user)

    def changelist_view(self, request, extra_context=None):
        response = super().changelist_view(
            request,
            extra_context=extra_context
        )
        
        try:
            # get only when times of days are set 
            qs = response.context_data['cl'].queryset.timesofdays()
        except (AttributeError, KeyError):
            return response

        response.context_data['scheduled_days'] = qs
        return response
