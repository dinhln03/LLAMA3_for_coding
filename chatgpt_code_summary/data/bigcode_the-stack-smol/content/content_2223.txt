from habit.habit_model import HabitHistory
from habit.complete_habit import complete


def test_overdue_habit(datasett):
    """
    please note the 'double tt' for datasett. This stands to differentiate
    the functional test data from the data used for unit tests.
    habit 1 is the overdue habit since its added first in the func/conftest
    module.
    :param datasett: from func/conftest
    :return:
    """
    session = datasett
    complete(1, session)
    result = session.query(HabitHistory.broken_count).\
        filter(HabitHistory.habitid == 1).all()
    assert result == [(1,)]


def test_a_habit_due_for_completion(datasett):
    """
    habit 2 is the due habit since its added second in the func/conftest
    module.
    :param datasett: from func/conftest
    :return:
    """
    session = datasett
    complete(2, session)
    result = session.query(HabitHistory.streak).\
        filter(HabitHistory.habitid == 2).all()
    assert result == [(1,)]
