##############################################################################
# Copyright 2019 Parker Berberian and Others                                 #
#                                                                            #
# Licensed under the Apache License, Version 2.0 (the "License");            #
# you may not use this file except in compliance with the License.           #
# You may obtain a copy of the License at                                    #
#                                                                            #
#    http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                            #
# Unless required by applicable law or agreed to in writing, software        #
# distributed under the License is distributed on an "AS IS" BASIS,          #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   #
# See the License for the specific language governing permissions and        #
# limitations under the License.                                             #
##############################################################################
from st2tests.base import BaseActionTestCase
from actions.actions import get_task_list
import json


class GetTaskListTestCase(BaseActionTestCase):
    action_cls = get_task_list.Task_List_Action

    def setUp(self):
        super(GetTaskListTestCase, self).setUp()
        self.action = self.get_action_instance()

    def test_tasklist_multiple_tasks(self):
        self.action.action_service.set_value("job_1", json.dumps({
            "access": {
                "task1": "asdf",
                "task2": "fdsa"
            }
        }), local=False)
        result = self.action.run(job_id=1, type="access")
        self.assertEqual(set(result), set(["task1", "task2"]))

    def test_tasklist_single_task(self):
        self.action.action_service.set_value("job_1", json.dumps({
            "access": {"task1": "asdf"},
            "hardware": {"task10": "asdf"}
        }), local=False)
        result = self.action.run(job_id=1, type="hardware")
        self.assertEqual(set(result), set(["task10"]))

    def test_empty_tasklist(self):
        self.action.action_service.set_value("job_1", json.dumps({
            "access": {"task1": "asdf"},
            "hardware": {"task10": "asdf"}
        }), local=False)
        result = self.action.run(job_id=1, type="unknown")
        self.assertFalse(result)
