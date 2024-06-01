# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A module to blocker devices based on device blocklists."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from tradefed_cluster import datastore_entities


def IsLabBlocked(lab_name):
  """Check if the lab is blocked.

  Args:
    lab_name: lab name
  Returns:
    true if the lab is blocked, otherwise false.
  """
  device_blocklists = (
      datastore_entities.DeviceBlocklist.query()
      .filter(datastore_entities.DeviceBlocklist.lab_name == lab_name)
      .fetch(1))
  return bool(device_blocklists)
