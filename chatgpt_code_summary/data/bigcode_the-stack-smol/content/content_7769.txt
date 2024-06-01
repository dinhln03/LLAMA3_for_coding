# Copyright (C) 2015 Google Inc., authors, and contributors <see AUTHORS file>
# Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>
# Created By: anze@reciprocitylabs.com
# Maintained By: anze@reciprocitylabs.com

"""Add finished/verified dates to cycle tasks

Revision ID: 13e52f6a9deb
Revises: 18bdb0671010
Create Date: 2016-01-04 13:52:43.017848

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '13e52f6a9deb'
down_revision = '18bdb0671010'

def upgrade():
  op.add_column('cycle_task_group_object_tasks', sa.Column('finished_date', sa.DateTime(), nullable=True))
  op.add_column('cycle_task_group_object_tasks', sa.Column('verified_date', sa.DateTime(), nullable=True))
  op.execute("""
      UPDATE cycle_task_group_object_tasks
      SET finished_date = updated_at
      WHERE status = "Finished"
  """)
  op.execute("""
      UPDATE cycle_task_group_object_tasks
      SET verified_date = updated_at, finished_date = updated_at
      WHERE status = "Verified"
  """)

def downgrade():
  op.drop_column('cycle_task_group_object_tasks', 'verified_date')
  op.drop_column('cycle_task_group_object_tasks', 'finished_date')
