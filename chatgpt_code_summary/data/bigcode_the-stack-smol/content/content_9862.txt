from django.apps import AppConfig


class BooksConfig(AppConfig):
    name = 'bookstudio.books'
    verbose_name = 'books'

    def ready(self):
        """Override this to put in:
            Users system checks
            Users signal registration
        """
        pass
