"""Module with functions around testing. You can run tests including doctest with generating coverage,
you can generate tests from readme or you can configure tests in conftest with single call."""

from mypythontools_cicd.tests.tests_internal import (
    add_readme_tests,
    deactivate_test_settings,
    default_test_config,
    run_tests,
    setup_tests,
    TestConfig,
)

__all__ = [
    "add_readme_tests",
    "deactivate_test_settings",
    "default_test_config",
    "run_tests",
    "setup_tests",
    "TestConfig",
]
