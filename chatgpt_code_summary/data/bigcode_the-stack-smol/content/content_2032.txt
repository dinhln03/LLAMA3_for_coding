# coding: utf-8
"""
Provides the exporter tool. The exporter can be used to export ComodIT entities
to local directories.
"""
from __future__ import print_function

from builtins import object
import os

from comodit_client.api.collection import EntityNotFoundException
from comodit_client.api.exceptions import PythonApiException
from comodit_client.api.host import Host
from comodit_client.rest.exceptions import ApiException
from comodit_client.util.path import ensure
import six
from comodit_client.api import orchestration


class ExportException(Exception):
    """
    Exception raised by exporter in case of error.
    """

    pass


class Export(object):
    """
    The exporter is a tool that enables to export entities to local
    directories. Exported entities may later be (re-)imported (see L{Import}).
    """

    def __init__(self, force = False):
        """
        Creates an exporter instance. If force flag is set, all data already
        present in a destination folder are overwritten on export.

        @param force: If True, force flag is set. It is not otherwise.
        @type force: bool
        """

        self._force = force

    def _export_files_content(self, entity, output_folder):
        for template in entity.files():
            file_name = template.name
            try:
                with open(os.path.join(output_folder, file_name), "w") as f:
                    if six.PY2:
                        f.write(template.read_content().encode('utf-8'))
                    else:
                        f.write(template.read_content())
            except ApiException as e:
                if e.code == 404:
                    pass
                else:
                    raise e

    def _export_entity(self, res, res_folder, export_files = False, export_thumb = False, backup = False):
        if backup:
            print("backup", res.name, "to", res_folder)
        else:
            print("exporting", res.name, "to", res_folder)

        # Ensures local repository does not contain stale data
        if(os.path.exists(res_folder) and len(os.listdir(res_folder)) > 0) and not self._force:
            raise ExportException(res_folder + " already exists and is not empty.")

        res.dump(res_folder)

        if export_files:
            # Dump files' content to disk
            files_folder = os.path.join(res_folder, "files")
            ensure(files_folder)
            self._export_files_content(res, files_folder)

        if export_thumb:
            # Dump thumbnail to disk
            try:
                content = res.read_thumbnail_content()
                with open(os.path.join(res_folder, "thumb"), "wb") as f:
                    f.write(content)
            except ApiException as e:
                if e.code == 404:
                    pass
                else:
                    raise e

    def export_application(self, app, path, backup = False):
        """
        Exports an application to a local folder.

        @param app: The application to export.
        @type app: L{Application}
        @param path: Path to local directory.
        @type path: string
        @param backup: indicate is a backup.
        @type path: bool
        """

        self._export_entity(app, path, True, True, backup)

    def export_distribution(self, dist, path, backup = False):
        """
        Exports a distribution to a local folder.

        @param dist: The distribution to export.
        @type dist: L{Distribution}
        @param path: Path to local directory.
        @type path: string
        @param backup: indicate is a backup.
        @type path: bool
        """

        self._export_entity(dist, path, True, True, backup)

    def export_platform(self, plat, path, backup = False):
        """
        Exports a platform to a local folder.

        @param plat: The platform to export.
        @type plat: L{Platform}
        @param path: Path to local directory.
        @type path: string
        @param backup: indicate is a backup.
        @type path: bool
        """

        self._export_entity(plat, path, True, backup=backup)

    def export_environment(self, env, path):
        """
        Exports an environment to a local folder. Hosts of the environment
        are exported also.

        @param env: The environment to export.
        @type env: L{Environment}
        @param path: Path to local directory.
        @type path: string
        """

        self._export_entity(env, path)

        hosts_folder = os.path.join(path, "hosts")
        for host in env.hosts():
            self.export_host(host, os.path.join(hosts_folder, host.name))
            
    def export_job(self, job, path):
        """
        Exports a job to a local folder.

        @param job: The job to export.
        @type job: L{Job}
        @param path: Path to local directory.
        @type path: string
        """

        self._export_entity(job, path)

    def export_orchestration(self, orchestration, path):
        """
        Exports a orchestration to a local folder.

        @param job: The orchestration to export.
        @type orchestration: L{Orchestration}
        @param path: Path to local directory.
        @type path: string
        """

        self._export_entity(orchestration, path)

    def export_notification(self, notification, path):
        """
        Exports a jobnotificationto a local folder.

        @param notification: The notification to export.
        @type notification: L{Notification}
        @param path: Path to local directory.
        @type path: string
        """

        self._export_entity(notification, path)

    def export_host(self, host, path):
        """
        Exports a host to a local folder. Contexts and instance are exported
        also.

        @param host: The host to export.
        @type host: L{Host}
        @param path: Path to local directory.
        @type path: string
        """

        self._export_entity(host, path)

        # Export instance
        if host.state != Host.State.DEFINED:
            try:
                instance = host.get_instance()
                instance.dump_json(os.path.join(path, "instance.json"))
            except PythonApiException:
                pass

        # Export application contexts
        app_folder = os.path.join(path, "applications")
        ensure(app_folder)
        for context in host.applications():
            context.dump_json(os.path.join(app_folder, context.application + ".json"))

        # Export platform context
        try:
            host.get_platform().dump_json(os.path.join(path, "platform.json"))
        except EntityNotFoundException:
            pass

        # Export distribution context
        try:
            host.get_distribution().dump_json(os.path.join(path, "distribution.json"))
        except EntityNotFoundException:
            pass

    def export_organization(self, org, path):
        """
        Exports an organization to a local folder. Environments, applications,
        distributions and platforms are exported also.

        @param org: The organization to export.
        @type org: L{Organization}
        @param path: Path to local directory.
        @type path: string
        """

        self._export_entity(org, path)

        for app in org.applications():
            self.export_application(app, os.path.join(path, "applications", app.name))

        for dist in org.distributions():
            self.export_distribution(dist, os.path.join(path, "distributions", dist.name))

        for plat in org.platforms():
            self.export_platform(plat, os.path.join(path, "platforms", plat.name))

        for job in org.jobs():
            self.export_job(job, os.path.join(path, "jobs", job.name))

        for orch in org.orchestrations():
            self.export_orchestration(orch, os.path.join(path, "orchestrations", orch.name))

        for env in org.environments():
            self.export_environment(env, os.path.join(path, "environments", env.name))