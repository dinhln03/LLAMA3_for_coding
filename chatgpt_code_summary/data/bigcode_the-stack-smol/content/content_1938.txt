# Copyright 2021 Arie Bregman
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
import crayons
import importlib
import logging
import os
import sys

from cinfo.config import Config
from cinfo.exceptions import usage as usage_exc

LOG = logging.getLogger(__name__)


class Triager(object):

    def __init__(self, config_file, source_name=None, target_name=None):
        self.config_file = config_file
        self.source_name = source_name
        self.target_name = target_name
        self.workspace = os.path.join(os.path.expanduser('~'), '.cinfo')

    def load_config(self):
        self.config = Config(file=self.config_file)
        self.config.load()
        self.sources = self.config.data['sources']
        self.targets = self.config.data['targets']

    def pull(self):
        LOG.info("{}: {}".format(
            crayons.yellow("pulling information from the source"),
            self.source_name))
        try:
            driver = getattr(importlib.import_module(
                "cinfo.drivers.{}".format(self.source['type'])),
                self.source['type'].capitalize())()
        except KeyError:
            LOG.error("{}: {}...exiting".format(
                crayons.red("No such source"), self.source))
            sys.exit(2)
        self.data = driver.pull(self.source['url'],
                                jobs=self.source['jobs'])
        if not self.data:
            LOG.warning("{}".format(crayons.red(
                "I've pulled nothing! outrageous!")))
        self.write(self.data)

    def publish(self):
        LOG.info("{}: {}".format(
            crayons.yellow("publishing data to target"),
            self.target['url']))
        try:
            publisher = getattr(importlib.import_module(
                "cinfo.drivers.{}".format(self.target['type'])),
                self.target['type'].capitalize())()
        except KeyError:
            LOG.error("{}: {}...exiting".format(
                crayons.red("No such target"), self.target))
            sys.exit(2)

        publisher.publish(self.data)

    def write(self, data):
        pass

    def validate(self):
        if len(self.sources.keys()) > 1 and not self.source_name:
            LOG.error(usage_exc.multiple_options("source"))
            sys.exit(2)
        elif not self.source_name:
            self.source = list(self.sources.values())[0]
        else:
            try:
                self.source = self.sources[self.source_name]
            except KeyError:
                LOG.error(usage_exc.missing_value(
                    self.source_name, [key for key in self.sources.keys()]))
                sys.exit(2)
        if len(self.targets.keys()) > 1 and not self.target:
            LOG.error(usage_exc.multiple_options("target"))
            sys.exit(2)
        elif not self.target_name:
            self.target = list(self.targets.values())[0]
        else:
            self.target = self.targets[self.target_name]

    def run(self):
        self.load_config()
        self.validate()
        self.pull()
        self.publish()
