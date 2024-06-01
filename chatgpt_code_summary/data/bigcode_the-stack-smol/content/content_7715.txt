'''A module for loading settings'''

import logging.config
import sys
from logging import getLogger
from pathlib import Path

import yaml

from hazelsync.metrics import get_metrics_engine

DEFAULT_SETTINGS = '/etc/hazelsync.yaml'
CLUSTER_DIRECTORY = '/etc/hazelsync.d'

DEFAULT_LOGGING = {
    'version': 1,
    'formatters': {
        'syslog': {'format': '%(name)s[%(process)d]: %(levelname)s: %(message)s'},
        'default': {'format': '%(asctime)s - %(name)s: %(levelname)s %(message)s'},
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'stream': sys.stderr,
            'formatter': 'default',
        },
        'syslog': {
            'level': 'INFO',
            'class': 'logging.handlers.SysLogHandler',
            'address': '/dev/log',
            'facility': 'local0',
            'formatter': 'syslog',
        },
    },
    'loggers': {
        'hazelsync': {
            'handlers': ['console', 'syslog'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}

log = getLogger('hazelsync')

class SettingError(AttributeError):
    '''Raise an exception if there is a configuration error'''
    def __init__(self, job, message):
        log.error("Configuration error (in %s): %s", job, message)
        super().__init__(message)

class GlobalSettings:
    '''A class to manage the global settings of hazelsync'''
    def __init__(self, path=DEFAULT_SETTINGS):
        path = Path(path)
        text = path.read_text(encoding='utf-8')
        data = yaml.safe_load(text)
        self.default_backend = data.get('default_backend', 'localfs')
        self.job_options = data.get('job_options')
        self.backend_options = data.get('backend_options')
        self.logging = data.get('logging', DEFAULT_LOGGING)

        metrics_config = data.get('metrics', {})
        self.metrics = get_metrics_engine(metrics_config)

    def logger(self):
        '''Setup logging and return the logger'''
        logging.config.dictConfig(self.logging)
        mylogger = getLogger('hazelsync')
        mylogger.debug('Logger initialized')
        return mylogger

    def job(self, job_type: str) -> dict:
        '''Return defaults for a job type'''
        return self.job_options.get(job_type, {})

    def backend(self, backend_type: str) -> dict:
        '''Return defaults for a backend type'''
        return self.backend_options.get(backend_type, {})

class ClusterSettings:
    '''A class to manage the settings of a cluster'''
    directory = Path(CLUSTER_DIRECTORY)

    def __init__(self, name, global_path=DEFAULT_SETTINGS):
        self.name = name
        self.globals = GlobalSettings(global_path)
        path = ClusterSettings.directory / f"{name}.yaml"
        text = path.read_text(encoding='utf-8')
        data = yaml.safe_load(text)

        self.job_type = data.get('job')
        self.job_options = data.get('options', {})
        self.backend_type = data.get('backend') or self.globals.default_backend
        self.backend_options = data.get('backend_options', {})

    @staticmethod
    def list() -> dict:
        '''List the backup cluster found in the settings'''
        settings = {}
        for path in ClusterSettings.directory.glob('*.yaml'):
            cluster = path.stem
            settings[cluster] = {'path': path}
            try:
                settings[cluster]['config_status'] = 'success'
            except KeyError as err:
                log.error(err)
                settings[cluster]['config'] = {}
                settings[cluster]['config_status'] = 'failure'
        return settings

    def job(self):
        '''Return the job options (merged with defaults)'''
        defaults = self.globals.job(self.job_type)
        options = self.job_options
        return self.job_type, {**defaults, **options}

    def backend(self):
        '''Return the backend option (merged with defaults)'''
        defaults = self.globals.backend(self.backend_type)
        options = self.backend_options
        return self.backend_type, {**defaults, **options}
