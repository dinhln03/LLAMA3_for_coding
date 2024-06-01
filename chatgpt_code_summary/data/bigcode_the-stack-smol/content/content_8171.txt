import datetime

import more_itertools

from annoworkapi.actual_working_time import (
    _create_actual_working_hours_dict,
    create_actual_working_times_daily,
    get_term_start_end_from_date_for_actual_working_time,
)

ACTUAL_WORKING_TIME_LIST = [
    {
        "job_id": "JOB_A",
        "workspace_member_id": "alice",
        "start_datetime": "2021-11-01T10:00:00.000Z",
        "end_datetime": "2021-11-01T11:00:00.000Z",
    },
    {
        "job_id": "JOB_B",
        "workspace_member_id": "alice",
        "start_datetime": "2021-11-01T12:00:00.000Z",
        "end_datetime": "2021-11-01T14:00:00.000Z",
    },
    {
        "job_id": "JOB_A",
        "workspace_member_id": "alice",
        "start_datetime": "2021-11-01T14:00:00.000Z",
        "end_datetime": "2021-11-01T18:00:00.000Z",
    },
]


class Test_get_term_start_end_from_date_for_actual_working_time:
    def test_with_jtc(self):
        actual = get_term_start_end_from_date_for_actual_working_time(
            "2021-04-01", "2021-04-01", tzinfo=datetime.timezone(datetime.timedelta(hours=9))
        )
        assert actual[0] == "2021-03-31T15:00:00.000Z"
        assert actual[1] == "2021-04-01T14:59:59.999Z"

    def test_with_utc(self):
        actual = get_term_start_end_from_date_for_actual_working_time(
            "2021-04-01", "2021-04-01", tzinfo=datetime.timezone.utc
        )
        assert actual[0] == "2021-04-01T00:00:00.000Z"
        assert actual[1] == "2021-04-01T23:59:59.999Z"


class Test__create_actual_working_hours_dict:
    jtc_tzinfo = datetime.timezone(datetime.timedelta(hours=9))

    def test_evening(self):
        actual = _create_actual_working_hours_dict(ACTUAL_WORKING_TIME_LIST[0], tzinfo=self.jtc_tzinfo)
        expected = {(datetime.date(2021, 11, 1), "alice", "JOB_A"): 1.0}
        assert actual == expected

    def test_midnight(self):
        actual = _create_actual_working_hours_dict(ACTUAL_WORKING_TIME_LIST[2], tzinfo=self.jtc_tzinfo)
        expected = {
            (datetime.date(2021, 11, 1), "alice", "JOB_A"): 1.0,
            (datetime.date(2021, 11, 2), "alice", "JOB_A"): 3.0,
        }
        assert actual == expected


class Test_create_actual_working_times_daily:
    jtc_tzinfo = datetime.timezone(datetime.timedelta(hours=9))

    def test_normal(self):
        actual = create_actual_working_times_daily(ACTUAL_WORKING_TIME_LIST, tzinfo=self.jtc_tzinfo)

        assert len(actual) == 3
        assert (
            more_itertools.first_true(actual, pred=lambda e: e["date"] == "2021-11-01" and e["job_id"] == "JOB_A")[
                "actual_working_hours"
            ]
            == 2.0
        )
        assert (
            more_itertools.first_true(actual, pred=lambda e: e["date"] == "2021-11-01" and e["job_id"] == "JOB_B")[
                "actual_working_hours"
            ]
            == 2.0
        )
        assert (
            more_itertools.first_true(actual, pred=lambda e: e["date"] == "2021-11-02" and e["job_id"] == "JOB_A")[
                "actual_working_hours"
            ]
            == 3.0
        )
