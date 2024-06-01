# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2015-2018 by ExopyHqcLegacy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the BSD license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Tests for the ApplyMagFieldTask

"""
from multiprocessing import Event

import pytest
import enaml

from exopy.tasks.api import RootTask
from exopy.tasks.tasks.logic.loop_task import LoopTask
from exopy.testing.util import show_and_close_widget
from exopy_hqc_legacy.tasks.tasks.instr.apply_mag_field_task\
    import ApplyMagFieldTask

with enaml.imports():
    from exopy.tasks.tasks.logic.views.loop_view import LoopView
    from exopy_hqc_legacy.tasks.tasks.instr.views.apply_mag_field_view\
        import ApplyMagFieldView

from .instr_helper import (InstrHelper, InstrHelperStarter, DummyJob,
                           PROFILES, DRIVERS)


class TestApplyMagFieldTask(object):

    def setup(self):
        self.root = RootTask(should_stop=Event(), should_pause=Event())
        self.task = ApplyMagFieldTask(name='Test',
                                      parallel={'activated': False})
        self.root.add_child_task(0, self.task)

        self.root.run_time[DRIVERS] = {'Test': (InstrHelper,
                                                InstrHelperStarter())}
        self.root.run_time[PROFILES] =\
            {'Test1':
                {'connections': {'C': {'owner': [],
                                       'output_fluctuations': 1e-6,
                                       'heater_state': []}},
                 'settings': {'S': {'sweep_to_field': [DummyJob(),  DummyJob(),
                                                       DummyJob()],
                                    'sweep_to_persistent_field': [DummyJob()],
                                    'read_persistent_field': [1],
                                    'check_connection': [True]}}
                 }
             }

        # This is set simply to make sure the test of InstrTask pass.
        self.task.selected_instrument = ('Test1', 'Test', 'C', 'S')

    def test_check1(self):
        """Simply test that everything is ok if field can be evaluated.

        """
        self.task.field = '3.0'

        test, traceback = self.task.check(test_instr=True)
        assert test
        assert not traceback

        assert self.task.get_from_database('Test_field') == 3.0

    def test_check2(self):
        """Check handling a wrong field.

        """
        self.task.field = '*1.0*'

        test, traceback = self.task.check(test_instr=True)
        assert not test
        assert len(traceback) == 1
        assert 'root/Test-field'in traceback

        assert self.task.get_from_database('Test_field') == 0.01

    def test_perform1(self):
        """Simple test when everything is right.

        """
        self.task.field = '2.0'

        self.root.prepare()
        self.task.perform()
        assert self.root.get_from_database('Test_field') == 2.0


@pytest.mark.ui
def test_apply_mag_field_view1(exopy_qtbot, root_view, task_workbench):
    """Test ApplyMagFieldView widget outisde of a LoopTask.

    """
    task = ApplyMagFieldTask(name='Test')
    root_view.task.add_child_task(0, task)
    show_and_close_widget(exopy_qtbot, ApplyMagFieldView(task=task, root=root_view))


@pytest.mark.ui
def test_apply_mag_field_view2(exopy_qtbot, root_view, task_workbench):
    """Test ApplyMagFieldView widget inside of a LoopTask.

    """
    task = ApplyMagFieldTask(name='Test')
    loop = LoopTask(name='r', task=task)
    root_view.task.add_child_task(0, loop)
    # XXX check for absence of target field
    show_and_close_widget(exopy_qtbot, LoopView(task=loop, root=root_view))
