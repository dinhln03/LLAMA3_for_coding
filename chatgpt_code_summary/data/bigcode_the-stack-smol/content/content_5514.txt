import importlib
import inspect
import logging
import os

from app.plugin import Hook, Command

PLUGINS_DIR = 'plugins'


def find_plugins():
    """Returns a list of plugin path names."""
    for root, dirs, files in os.walk(PLUGINS_DIR):
        for file in files:
            if file.endswith('.py'):
                yield os.path.join(root, file)


def load_plugins(hook_plugins, command_plugins):
    """Populates the plugin lists."""
    for file in find_plugins():
        try:
            module_name = os.path.splitext(os.path.basename(file))[0]
            module = importlib.import_module(PLUGINS_DIR + '.' + module_name)
            for entry_name in dir(module):
                entry = getattr(module, entry_name)
                if not inspect.isclass(entry) or inspect.getmodule(entry) != module:
                    continue
                if issubclass(entry, Hook):
                    hook_plugins.append(entry())
                elif issubclass(entry, Command):
                    command_plugins.append(entry())
        except (ImportError, NotImplementedError):
            continue


def process_commands(input_obj, commands):
    logging.debug('Processing commands')

    hook_plugins = []
    command_plugins = []
    load_plugins(hook_plugins, command_plugins)

    for command_str in commands:
        for plugin in command_plugins:
            if command_str in plugin.names:
                for hook in hook_plugins:
                    hook.before_handle(input_obj, plugin)
                input_obj = plugin.handle(input_obj)
                for hook in hook_plugins:
                    hook.after_handle(input_obj, plugin)
