import pytest

from exchange_calendars.exchange_calendar_xshg import XSHGExchangeCalendar
from .test_exchange_calendar import ExchangeCalendarTestBase
from .test_utils import T


class TestXSHGCalendar(ExchangeCalendarTestBase):
    @pytest.fixture(scope="class")
    def calendar_cls(self):
        yield XSHGExchangeCalendar

    @pytest.fixture
    def max_session_hours(self):
        # Shanghai stock exchange is open from 9:30 am to 3pm
        yield 5.5

    @pytest.fixture
    def start_bound(self):
        yield T("1999-01-01")

    @pytest.fixture
    def end_bound(self):
        yield T("2025-12-31")

    @pytest.fixture
    def regular_holidays_sample(self):
        yield [
            # 2017
            "2017-01-02",
            "2017-01-27",
            "2017-01-30",
            "2017-01-31",
            "2017-02-01",
            "2017-02-02",
            "2017-04-03",
            "2017-04-04",
            "2017-05-01",
            "2017-05-29",
            "2017-05-30",
            "2017-10-02",
            "2017-10-03",
            "2017-10-04",
            "2017-10-05",
            "2017-10-06",

            # 2020
            "2020-01-31"
        ]
