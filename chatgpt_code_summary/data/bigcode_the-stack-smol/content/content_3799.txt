import requests
import json
import datetime
import os
import io

from invoke import task

from .invoke_utils import ServerConnection, use_dump_modifier_function

RAJK_PASSWORD = os.environ.get("RAJK_PASSWORD")
RAJK_RSA = os.environ.get("RAJK_RSA")

TEST_DEPLOY_DIRECTORY = os.getcwd() + "/build"

rajk_server_connection = ServerConnection(
    "rajk", "146.110.60.20", 2222, "/var/www/rajkdjango2/bin/python"
)


def redo_rsa_from_text(c, rsa_text):
    os.makedirs("{}/.ssh".format(os.path.expanduser("~")), exist_ok=True)
    rsa_path = "{}/.ssh/id_rsa".format(os.path.expanduser("~"))

    with open(rsa_path, "w") as fp:
        fp.write(rsa_text)

    c.run("chmod 600 {}".format(rsa_path))

@task
def backup_django(c):
    os.makedirs("backups", exist_ok=True)
    bup_dir = os.path.join("backups", datetime.date.today().isoformat())
    c.run("mkdir {}".format(bup_dir))
    scp_command = rajk_server_connection.copy_from_server_command(
        bup_dir, "/var/www/rajkdjango2"
    )
    c.run(scp_command)


@task
def restart_server(c):
    command = rajk_server_connection.run_sudo_command(
        "service django2 restart", RAJK_PASSWORD
    )
    c.run(command)


@task
def stop_server(c):
    command = rajk_server_connection.run_sudo_command(
        "service django2 stop", RAJK_PASSWORD
    )
    c.run(command)


@task
def start_server(c):
    command = rajk_server_connection.run_sudo_command(
        "service django2 start", RAJK_PASSWORD
    )
    c.run(command)


@task
def dump(c, fname="dump.json", no_contenttypes=False):
    py_command = "/var/www/rajkdjango2/manage.py dumpdata {}".format(
        "-e contenttypes" if no_contenttypes else ""
    )

    command = rajk_server_connection.remote_python_command(py_command)
    c.run(command + " > {}".format(fname))


@task
def remote_dump(c, no_contenttypes=True):
    py_command = "/var/www/rajkdjango2/manage.py dumpdata {} > /var/www/rajk/djangodump.json".format(
        "-e contenttypes" if no_contenttypes else ""
    )

    command = rajk_server_connection.remote_python_command(py_command)
    c.run(command)


@task
def setup_test_deploy_env(c):

    c.run("rm -rf ./{}".format(TEST_DEPLOY_DIRECTORY))
    c.run("mkdir {}".format(TEST_DEPLOY_DIRECTORY))

    resp = requests.get("https://api.github.com/orgs/rajk-apps/repos")
    repos = [
        "git+https://github.com/{}".format(d["full_name"])
        for d in json.loads(resp.content)
    ]
    app_names = [r.split("/")[-1].replace("-", "_") for r in repos]

    c.run("python3 -m venv {}/django_venv".format(TEST_DEPLOY_DIRECTORY))
    for r in ["wheel", "django", "toml"] + repos:
        c.run("{}/django_venv/bin/pip install {}".format(TEST_DEPLOY_DIRECTORY, r))

    c.run(
        "cd {};django_venv/bin/django-admin startproject rajkproject".format(
            TEST_DEPLOY_DIRECTORY
        )
    )

    with open(
        "{}/rajkproject/rajkproject/settings.py".format(TEST_DEPLOY_DIRECTORY), "a"
    ) as fp:
        fp.write(
            "\nINSTALLED_APPS += [{}]".format(
                ", ".join(["'{}'".format(a) for a in app_names])
            )
        )

    with open(
        "{}/rajkproject/rajkproject/urls.py".format(TEST_DEPLOY_DIRECTORY), "a"
    ) as fp:
        fp.write(
            "\nfrom django.urls import include"
            "\nurlpatterns.append(path('accounts/', include('django.contrib.auth.urls')))"
            "\nurlpatterns += [{}]".format(
                ", ".join(
                    [
                        "path('{}', include('{}.urls'))".format(
                            a + "/" if a != "rajk_appman" else "", a
                        )
                        for a in app_names
                    ]
                )
            )
        )

    dump_fname = "{}/dump.json".format(TEST_DEPLOY_DIRECTORY)

    resp = requests.get("https://rajk.uni-corvinus.hu/djangodump.json")

    with open(dump_fname, "wb") as fp:
        fp.write(resp.content)

    for django_command in [
        "makemigrations",
        "makemigrations {}".format(" ".join(app_names)),
        "migrate",
        "loaddata {}".format(dump_fname),
    ]:
        c.run(
            "{}/django_venv/bin/python {}/rajkproject/manage.py {}".format(
                TEST_DEPLOY_DIRECTORY, TEST_DEPLOY_DIRECTORY, django_command
            )
        )


@task
def deploy(c, dump_modifier_function=None, live=False, redo_rsa=False):

    f = io.StringIO()
    c.run(
        "{}/django_venv/bin/python setup.py --fullname".format(TEST_DEPLOY_DIRECTORY),
        out_stream=f,
    )
    current_app_fullname = f.getvalue().strip()
    f.close()
    c.run("{}/django_venv/bin/python setup.py sdist".format(TEST_DEPLOY_DIRECTORY))

    local_tarball = "./dist/{}.tar.gz".format(current_app_fullname)

    c.run(
        "{}/django_venv/bin/pip install {}".format(TEST_DEPLOY_DIRECTORY, local_tarball)
    )

    dump_fname = "{}/dump.json".format(TEST_DEPLOY_DIRECTORY)

    resp = requests.get("https://rajk.uni-corvinus.hu/djangodump.json")
    with open(dump_fname, "wb") as fp:
        fp.write(resp.content)

    if dump_modifier_function is not None:
        use_dump_modifier_function(dump_modifier_function, dump_fname)

    c.run("rm {}/rajkproject/db.sqlite3".format(TEST_DEPLOY_DIRECTORY))

    for django_command in [
        "makemigrations",
        "makemigrations {}".format(current_app_fullname.split("-")[0]),
        "migrate",
        "loaddata {}".format(dump_fname)
    ]:
        c.run(
            "{}/django_venv/bin/python {}/rajkproject/manage.py {}".format(
                TEST_DEPLOY_DIRECTORY, TEST_DEPLOY_DIRECTORY, django_command
            )
        )

    if live:
        _live_deploy(c, local_tarball, current_app_fullname, dump_modifier_function, redo_rsa)


def _live_deploy(c, local_tarball, current_app_fullname, dump_modifier_function=None, redo_rsa=False):

    if redo_rsa:
        if RAJK_RSA:
            redo_rsa_from_text(c, RAJK_RSA)
        else:
            raise EnvironmentError("No RAJK_RSA env variable")

    local_dump_fname = "{}/deploy_dump.json".format(TEST_DEPLOY_DIRECTORY)
    remote_dump_fname = "/var/www/rajkdjango2/deploy_dump.json"

    print("stopping server")
    stop_server(c)
    print("dumping data")
    dump(c, local_dump_fname, True)

    if dump_modifier_function is not None:
        use_dump_modifier_function(dump_modifier_function, local_dump_fname)

    scp_command = rajk_server_connection.copy_to_server_command(
        local_dump_fname, remote_dump_fname
    )
    c.run(scp_command)

    remote_tarball = "/var/www/rajkdjango2/tarballs/{}".format(
        local_tarball.split("/")[-1]
    )

    tar_scp_command = rajk_server_connection.copy_to_server_command(
        local_tarball, remote_tarball
    )
    c.run(tar_scp_command)
    install_command = "/var/www/rajkdjango2/bin/pip --no-cache-dir install --upgrade {}".format(
        remote_tarball
    )
    remote_install_command = rajk_server_connection.run_ssh_command(install_command)

    c.run(remote_install_command)
    c.run(rajk_server_connection.run_ssh_command("rm /var/www/rajkdjango2/db.sqlite3"))

    for django_command in [
        "makemigrations",
        "makemigrations {}".format(current_app_fullname.split("-")[0]),
        "migrate",
        "loaddata {}".format(remote_dump_fname),
    ]:
        c.run(
            rajk_server_connection.remote_python_command(
                "/var/www/rajkdjango2/manage.py {}".format(django_command)
            )
        )

    start_server(c)
