#!/usr/bin/env python
#
# Copyright (c) 2018 All rights reserved
# This program and the accompanying materials
# are made available under the terms of the Apache License, Version 2.0
# which accompanies this distribution, and is available at
#
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Define the classes required to fully cover k8s."""

import logging
import os
import unittest


from security_tests import SecurityTesting


class SecurityTests(unittest.TestCase):

    # pylint: disable=missing-docstring

    def setUp(self):
        os.environ["DEPLOY_SCENARIO"] = "k8-test"
        os.environ["KUBE_MASTER_IP"] = "127.0.0.1"
        os.environ["KUBE_MASTER_URL"] = "https://127.0.0.1:6443"
        os.environ["KUBERNETES_PROVIDER"] = "local"

        self.security_stesting = SecurityTesting.SecurityTesting()

    def test_run_kubetest_cmd_none(self):
        with self.assertRaises(TypeError):
            self.security_stesting.run_security()


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
