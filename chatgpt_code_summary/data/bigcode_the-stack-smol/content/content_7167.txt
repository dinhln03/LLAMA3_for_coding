import os

import fabdeploytools.envs
from fabric.api import env, lcd, local, task
from fabdeploytools import helpers

import deploysettings as settings

env.key_filename = settings.SSH_KEY
fabdeploytools.envs.loadenv(settings.CLUSTER)

ROOT, PROJECT_NAME = helpers.get_app_dirs(__file__)


@task
def build():
    with lcd(PROJECT_NAME):
        local('npm install')
        local('make install')
        local('cp src/media/js/settings_local_hosted.js '
              'src/media/js/settings_local.js')
        local('make build')
        local('node_modules/.bin/commonplace langpacks')


@task
def deploy_jenkins():
    r = helpers.build_rpm(name=settings.PROJECT_NAME,
                          app_dir='marketplace-style-guide',
                          env=settings.ENV,
                          cluster=settings.CLUSTER,
                          domain=settings.DOMAIN,
                          root=ROOT)

    r.local_install()
    r.remote_install(['web'])
