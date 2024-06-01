#!/usr/bin/env python3

import uuid
import random
import datetime
from faker import Factory
fake = Factory.create()

num_people = 1000

last_jump_start = datetime.datetime(2008, 9, 1)
last_jump_end = datetime.datetime(2016, 8, 1)

print('COPY members (uuid, name, email, phone_number, last_jump, created_at, updated_at) FROM stdin;')

for i in range(0, num_people):
  member_uuid = str(uuid.uuid4())
  name = fake.name()
  email = fake.email()
  phone_number = '+447' + str(random.randrange(100000000, 999999999, 1))
  last_jump = fake.date_time_between_dates(datetime_start = last_jump_start, datetime_end = last_jump_end).strftime('%Y-%m-%d')
  created_at = fake.date_time_between_dates(datetime_start = last_jump_start, datetime_end = last_jump_end)
  updated_at = fake.date_time_between_dates(datetime_start = created_at, datetime_end = last_jump_end).strftime('%Y-%m-%d %H:%M:%S')

  print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (member_uuid, name, email, phone_number, last_jump, created_at.strftime('%Y-%m-%d %H:%M:%S'), updated_at))

print('\\.')

