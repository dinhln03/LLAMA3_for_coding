from django.contrib.staticfiles.storage import staticfiles_storage
from django.urls import reverse
from ManagementStudents.jinja2 import Environment


# This enables us to use Django template tags like {% url ‘index’ %} or {% static ‘path/to/static/file.js’ %} in our Jinja2 templates.
def environment(**options):
    env = Environment(**options)
    env.globals.update({
        'static': staticfiles_storage.url,
        'url': reverse,
    })
    return env
