import django
import six
from django.http import HttpResponseRedirect

if django.VERSION[0] < 2:
    from django.core.urlresolvers import reverse
else:
    from django.urls import reverse

from django.db import transaction
from django.utils import timezone

import logging
from processlib.assignment import inherit
from processlib.tasks import run_async_activity


logger = logging.getLogger(__name__)


@six.python_2_unicode_compatible
class Activity(object):
    def __init__(
        self,
        flow,
        process,
        instance,
        name,
        verbose_name=None,
        permission=None,
        auto_create_permission=True,
        permission_name=None,
        skip_if=None,
        assign_to=inherit,
    ):
        self.flow = flow
        self.process = process
        self.verbose_name = verbose_name
        self.permission = permission
        self.auto_create_permission = auto_create_permission
        self.permission_name = permission_name or verbose_name or name
        self.name = name
        self.instance = instance

        # ensure that we have a single referenced process object
        if self.instance:
            self.instance.process = self.process

        self._skip = skip_if
        self._get_assignment = assign_to

    def should_skip(self):
        if not self._skip:
            return False
        return self._skip(self)

    def should_wait(self):
        return False

    def has_view(self):
        return False

    def __str__(self):
        return six.text_type(self.verbose_name or self.name)

    def __repr__(self):
        return '{}(name="{}")'.format(self.__class__.__name__, self.name)

    def instantiate(
        self, predecessor=None, instance_kwargs=None, request=None, **kwargs
    ):
        assert not self.instance
        instance_kwargs = instance_kwargs or {}

        request_user = (
            request.user if request and request.user.is_authenticated else None
        )
        user, group = self._get_assignment(
            request_user=request_user, predecessor=predecessor
        )

        if "assigned_user" not in instance_kwargs:
            instance_kwargs["assigned_user"] = user
        if "assigned_group" not in instance_kwargs:
            instance_kwargs["assigned_group"] = group

        self.instance = self.flow.activity_model(
            process=self.process, activity_name=self.name, **(instance_kwargs or {})
        )
        self.instance.save()
        if predecessor:
            self.instance.predecessors.add(predecessor.instance)

    def assign_to(self, user, group):
        self.instance.assigned_user = user
        self.instance.assigned_group = group
        self.instance.save()

    def start(self, **kwargs):
        assert self.instance.status in (
            self.instance.STATUS_INSTANTIATED,
            self.instance.STATUS_SCHEDULED,
        )
        if not self.instance.started_at:
            self.instance.started_at = timezone.now()
        self.instance.status = self.instance.STATUS_STARTED

    def finish(self, **kwargs):
        assert self.instance.status == self.instance.STATUS_STARTED
        if not self.instance.finished_at:
            self.instance.finished_at = timezone.now()
        self.instance.status = self.instance.STATUS_DONE
        self.instance.modified_by = kwargs.get("user", None)
        self.instance.save()
        self._instantiate_next_activities()

    def cancel(self, **kwargs):
        assert self.instance.status in (
            self.instance.STATUS_INSTANTIATED,
            self.instance.STATUS_ERROR,
        )
        self.instance.status = self.instance.STATUS_CANCELED
        self.instance.modified_by = kwargs.get("user", None)
        self.instance.save()

    def undo(self, **kwargs):
        assert self.instance.status == self.instance.STATUS_DONE
        self.instance.finished_at = None
        self.instance.status = self.instance.STATUS_INSTANTIATED
        self.instance.modified_by = kwargs.get("user", None)
        self.instance.save()

        undo_callback = getattr(self.process, "undo_{}".format(self.name), None)
        if undo_callback is not None:
            undo_callback()

    def error(self, **kwargs):
        assert self.instance.status != self.instance.STATUS_DONE
        self.instance.status = self.instance.STATUS_ERROR
        self.instance.finished_at = timezone.now()
        self.instance.modified_by = kwargs.get("user", None)
        self.instance.save()

    def _get_next_activities(self):
        for activity_name in self.flow._out_edges[self.name]:
            activity = self.flow._get_activity_by_name(
                process=self.process, activity_name=activity_name
            )
            if activity.should_skip():
                for later_activity in activity._get_next_activities():
                    yield later_activity
            else:
                yield activity

    def _instantiate_next_activities(self):
        for activity in self._get_next_activities():
            activity.instantiate(predecessor=self)


class State(Activity):
    """
    An activity that simple serves as a marker for a certain state being reached, e.g.
    if the activity before it was conditional.
    """

    def instantiate(self, **kwargs):
        super(State, self).instantiate(**kwargs)
        self.start()
        self.finish()


class ViewActivity(Activity):
    def __init__(self, view=None, **kwargs):
        super(ViewActivity, self).__init__(**kwargs)
        if view is None:
            raise ValueError(
                "A ViewActivity requires a view, non given for {}.{}".format(
                    self.flow.label, self.name
                )
            )
        self.view = view

    def has_view(self):
        return True

    def get_absolute_url(self):
        return reverse(
            "processlib:process-activity",
            kwargs={"flow_label": self.flow.label, "activity_id": self.instance.pk},
        )

    def dispatch(self, request, *args, **kwargs):
        kwargs["activity"] = self
        return self.view(request, *args, **kwargs)


class FunctionActivity(Activity):
    def __init__(self, callback=None, **kwargs):
        self.callback = callback
        super(FunctionActivity, self).__init__(**kwargs)

    def instantiate(self, **kwargs):
        super(FunctionActivity, self).instantiate(**kwargs)
        self.start()

    def start(self, **kwargs):
        super(FunctionActivity, self).start(**kwargs)

        try:
            self.callback(self)
        except Exception as e:
            logger.exception(e)
            self.error(exception=e)
            return

        self.finish()

    def retry(self):
        assert self.instance.status == self.instance.STATUS_ERROR
        self.instance.status = self.instance.STATUS_INSTANTIATED
        self.instance.finished_at = None
        self.instance.save()
        self.start()


class AsyncActivity(Activity):
    def __init__(self, callback=None, **kwargs):
        self.callback = callback
        super(AsyncActivity, self).__init__(**kwargs)

    def instantiate(self, **kwargs):
        super(AsyncActivity, self).instantiate(**kwargs)
        self.schedule()

    def schedule(self, **kwargs):
        self.instance.status = self.instance.STATUS_SCHEDULED
        self.instance.scheduled_at = timezone.now()
        self.instance.save()
        transaction.on_commit(
            lambda: run_async_activity.delay(self.flow.label, self.instance.pk)
        )

    def retry(self, **kwargs):
        assert self.instance.status == self.instance.STATUS_ERROR
        self.instance.status = self.instance.STATUS_INSTANTIATED
        self.instance.finished_at = None
        self.schedule(**kwargs)

    def start(self, **kwargs):
        super(AsyncActivity, self).start(**kwargs)
        self.callback(self)


class AsyncViewActivity(AsyncActivity):
    """
    An async activity that renders a view while the async task is running.
    The view could be AsyncActivityView with a custom template_name
    """

    def __init__(self, view=None, **kwargs):
        super(AsyncViewActivity, self).__init__(**kwargs)
        if view is None:
            raise ValueError(
                "An AsyncViewActivity requires a view, non given for {}.{}".format(
                    self.flow.label, self.name
                )
            )
        self.view = view

    def has_view(self):
        return True

    def get_absolute_url(self):
        return reverse(
            "processlib:process-activity",
            kwargs={"flow_label": self.flow.label, "activity_id": self.instance.pk},
        )

    def dispatch(self, request, *args, **kwargs):
        kwargs["activity"] = self
        return self.view(request, *args, **kwargs)


class StartMixin(Activity):
    def instantiate(
        self, predecessor=None, instance_kwargs=None, request=None, **kwargs
    ):
        assert not self.instance
        assert not predecessor
        instance_kwargs = instance_kwargs or {}

        request_user = (
            request.user if request and request.user.is_authenticated else None
        )
        user, group = self._get_assignment(
            request_user=request_user, predecessor=predecessor
        )
        if "assigned_user" not in instance_kwargs:
            instance_kwargs["assigned_user"] = user
        if "assigned_group" not in instance_kwargs:
            instance_kwargs["assigned_group"] = group

        self.instance = self.flow.activity_model(
            process=self.process, activity_name=self.name, **(instance_kwargs or {})
        )

    def finish(self, **kwargs):
        assert self.instance.status == self.instance.STATUS_STARTED
        if not self.instance.finished_at:
            self.instance.finished_at = timezone.now()

        self.process.save()
        self.instance.process = self.process
        self.instance.status = self.instance.STATUS_DONE
        self.instance.modified_by = kwargs.get("user", None)
        self.instance.save()
        self._instantiate_next_activities()


class StartActivity(StartMixin, Activity):
    pass


class StartViewActivity(StartMixin, ViewActivity):
    pass


class EndActivity(Activity):
    def instantiate(self, **kwargs):
        super(EndActivity, self).instantiate(**kwargs)
        self.start()
        self.finish()

    def finish(self, **kwargs):
        super(EndActivity, self).finish(**kwargs)

        update_fields = []
        if not self.process.finished_at:
            self.process.finished_at = self.instance.finished_at
            update_fields.append("finished_at")

        if not self.process.status == self.process.STATUS_DONE:
            self.process.status = self.process.STATUS_DONE
            update_fields.append("status")

        self.process.save(update_fields=update_fields)


class EndRedirectActivity(EndActivity):
    def __init__(self, redirect_url_callback=None, **kwargs):
        self.redirect_url_callback = redirect_url_callback
        super(EndActivity, self).__init__(**kwargs)

    def instantiate(self, **kwargs):
        # HACK: we skip the EndActivity implementation
        # because it would finish the activity right away
        super(EndActivity, self).instantiate(**kwargs)

    def has_view(self):
        return True

    def get_absolute_url(self):
        return reverse(
            "processlib:process-activity",
            kwargs={"flow_label": self.flow.label, "activity_id": self.instance.pk},
        )

    def dispatch(self, request, *args, **kwargs):
        self.start()
        url = reverse(
            "processlib:process-detail", kwargs={"pk": self.instance.process.pk}
        )
        try:
            if self.redirect_url_callback:
                url = self.redirect_url_callback(self)
            self.finish()
        except Exception as e:
            logger.exception(e)
            self.error(exception=e)
        return HttpResponseRedirect(url)


class FormActivity(Activity):
    def __init__(self, form_class=None, **kwargs):
        self.form_class = form_class
        super(FormActivity, self).__init__(**kwargs)

    def get_form(self, **kwargs):
        return self.form_class(**kwargs)


class StartFormActivity(StartMixin, FormActivity):
    pass


class IfElse(Activity):
    def __init__(self, flow, process, instance, name, **kwargs):
        super(IfElse, self).__init__(flow, process, instance, name, **kwargs)


class Wait(Activity):
    def __init__(self, flow, process, instance, name, **kwargs):
        wait_for = kwargs.pop("wait_for", None)
        if not wait_for:
            raise ValueError("Wait activity needs to wait for something.")

        super(Wait, self).__init__(flow, process, instance, name, **kwargs)

        self._wait_for = set(wait_for) if wait_for else None

    def _find_existing_instance(self, predecessor):
        candidates = list(
            self.flow.activity_model.objects.filter(
                process=self.process, activity_name=self.name
            )
        )

        for candidate in candidates:
            # FIXME this only corrects for simple loops, may fail with more complex scenarios
            if not candidate.successors.filter(
                status=candidate.STATUS_DONE, activity_name=self.name
            ).exists():
                return candidate

        raise self.flow.activity_model.DoesNotExist()

    def instantiate(self, predecessor=None, instance_kwargs=None, **kwargs):
        if predecessor is None:
            raise ValueError("Can't wait for something without a predecessor.")

        # find the instance
        try:
            self.instance = self._find_existing_instance(predecessor)
        except self.flow.activity_model.DoesNotExist:
            self.instance = self.flow.activity_model(
                process=self.process, activity_name=self.name, **(instance_kwargs or {})
            )
            self.instance.save()

        self.instance.predecessors.add(predecessor.instance)
        self.start()

    def start(self, **kwargs):
        if not self.instance.started_at:
            self.instance.started_at = timezone.now()

        self.instance.status = self.instance.STATUS_STARTED
        self.instance.save()

        predecessor_names = {
            instance.activity_name for instance in self.instance.predecessors.all()
        }
        if self._wait_for.issubset(predecessor_names):
            self.finish()
