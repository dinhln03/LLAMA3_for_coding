from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader


class _MutatioEnvironmentMeta(type):
    """Collects the mutatio tags into Mongo for the admin page."""
    def __init__(cls, name, bases, dct):
        super(_MutatioEnvironmentMeta, cls).__init__(name, bases, dct)


class MutatioEnvironment(Environment, metaclass=_MutatioEnvironmentMeta):
    def __init__(self, *args, **kwargs):
        self.app = kwargs.pop('app', None)
        super(MutatioEnvironment, self).__init__(*args, **kwargs)


class MutatioFileSystemLoader(FileSystemLoader):
    def __init__(self, *args, **kwargs):
        super(MutatioFileSystemLoader, self).__init__(*args, **kwargs)
        # This is too rigid
        self.create_tags(args[0])

    def create_tags(self):
        print("Tags creating......")
