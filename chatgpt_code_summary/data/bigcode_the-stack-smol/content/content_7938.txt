import glob
import os
from distutils.dir_util import copy_tree
from shutil import copy

from alembic.command import init as _init

from metric.cli.conf import Conf
from metric.cli.template import Template
from metric.src import Base
from metric.src.package import Package


def init(name):
    """
    ## Init

    [ID]
        Init adalah perintah inisiasi oleh metric untuk membuat sebuah project dengan pondasi yang telah di setup, cara
        penggunaan ini bisa dengan 2 cara, membuat project dari direktori saat ini (CWD) atau dengan direktori baru.
    [EN]
        Init is the command initiation by metric to create a project with the foundation that has been setup, there are
        2 ways to work with it, either you can create from current working directory (CWD) or new directory.

    @param name: project name
    """
    project_path = os.getcwd()

    if name != '.':
        project_path = os.path.join(os.getcwd(), name)
        Package.make_directory(project_path)

    _init(Base.base_configuration(project_path), project_path)

    packages_build = {
        'apps': ('resources',),
        'models': tuple(),
    }

    for k, v in packages_build.items():
        Package.make_package(os.path.join(project_path, k))

        for i in v:
            Package.make_package(os.path.join(project_path, k, i))

    dir_build = {
        'apps': ('views',),
        'models': ('fields',),
        '.': ('scripts',)
    }

    for k, v in dir_build.items():
        for i in v:
            Package.make_directory(os.path.join(project_path, k, i))

    file_remove = ['script.py.mako']
    [os.remove(os.path.join(project_path, i)) for i in file_remove]

    scripts = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../scripts")
    [copy(file, os.path.join(project_path, 'scripts')) for file in glob.glob(os.path.join(scripts, "*.mako"))]

    os.rename(os.path.join(project_path, 'env.py'), os.path.join(project_path, 'scripts', 'env.py'))

    copy_tree(os.path.join(scripts, "setup"), project_path)
    for file in glob.glob(os.path.join(os.path.join(scripts, "setup"), "*.py")):
        copy(file, project_path)

    Conf.reset(project_path)


def make_resource(name):
    """
    ## Make resource

    [ID]
        Perintah ini adalah suatu perintah yang digunakan untuk membuat "resource" baru dari "script" yang telah di
        sediakan.
    [EN]
        This is a command that used to create new "resource" based from the existing "script" provided.

    @param name: resource name
    """
    t = Template()
    t.template_type = 'resource'
    t.make(name)
