from unittest.mock import Mock
from pysync_redmine.repositories.redmine import RedmineRepo
from pysync_redmine.domain import (
                                   Project,
                                   Task,
                                   Phase,
                                   Member,
                                   Calendar
                                   )
import datetime


def get_basic_frame(project):
        mock_repository = Mock()
        project.repository = mock_repository

        phase = Phase(project)
        phase._id = 2
        phase.save()

        member = Member(project, 'member_key')
        member._id = 5
        member.save()

        main_task = Task(project)
        main_task.description = 'Initial description'
        main_task.start_date = datetime.date(2016, 1, 4)
        main_task.duration = 2
        main_task.complete = 75
        main_task._id = 1
        main_task.save()

        parent = Task(project)
        parent.description = 'parent description'
        parent.start_date = datetime.date(2016, 1, 4)
        parent.duration = 1
        parent.complete = 100
        parent._id = 2
        parent.save()

        next_task = Task(project)
        next_task.description = 'next_task description'
        next_task.start_date = datetime.date(2016, 1, 4)
        next_task.duration = 1
        next_task.complete = 100
        next_task._id = 3
        next_task.save()

        return (phase, member, parent, main_task, next_task)
