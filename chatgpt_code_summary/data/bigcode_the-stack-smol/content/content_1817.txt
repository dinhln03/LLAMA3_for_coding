""" Orlov Module : workspace module fixture. """
import os
import logging

import pytest
from orlov.libs.workspace import Workspace

logger = logging.getLogger(__name__)


@pytest.fixture(scope='session')
def workspace(request) -> Workspace:
    """ Workspace Factory Fixture.

    Yields:
        directory(Workspace): Workspace Created.

    """
    logger.debug('Setup of test structure.')
    # create screenshot directory
    if request.config.getoption('workspace'):
        result_dir = request.config.getoption('workspace')
    else:
        if not os.path.exists('result'):
            logger.debug('Creating results folder to store results')
            os.mkdir('result')
        result_dir = os.path.join(os.getcwd(), 'result')
    logger.debug('Created folder %s', result_dir)
    yield Workspace(result_dir)
