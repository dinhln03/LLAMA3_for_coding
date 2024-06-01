from unittest.mock import MagicMock, patch, call
from tagtrain import data
from . import fake

from tagtrain.tagtrain.tt_remove import Remove


@patch('tagtrain.data.by_owner.remove_user_from_group')
def test_unknown_group(remove_user_from_group):
    remove_user_from_group.side_effect = data.Group.DoesNotExist()

    app, reply, message, match = fake.create_all()

    Remove(app).run(reply, message, match)

    remove_user_from_group.assert_called_once_with('AuthorName', 'GroupName', 'MemberName')
    reply.append.assert_called_once_with('Group `GroupName` does not exist.  Skipping.')


@patch('tagtrain.data.by_owner.remove_user_from_group')
def test_unknown_member(remove_user_from_group):
    remove_user_from_group.side_effect = data.Member.DoesNotExist()

    app, reply, message, match = fake.create_all()

    Remove(app).run(reply, message, match)

    remove_user_from_group.assert_called_once_with('AuthorName', 'GroupName', 'MemberName')
    reply.append.assert_called_once_with('`MemberName` is not a Member of Group `GroupName`.  Skipping.')

@patch('tagtrain.data.by_owner.remove_user_from_group')
def test_good(remove_user_from_group):
    remove_user_from_group.return_value = fake.create_group(name='GroupName', member_count=99)

    app, reply, message, match = fake.create_all()

    Remove(app).run(reply, message, match)

    remove_user_from_group.assert_called_once_with('AuthorName', 'GroupName', 'MemberName')
    reply.append.assert_called_once_with('`MemberName` removed from Group `GroupName`, 99 total Members.')


@patch('tagtrain.data.by_owner.remove_user_from_group')
def test_good_no_members(remove_user_from_group):
    remove_user_from_group.return_value = fake.create_group(name='GroupName', member_count=0)

    app, reply, message, match = fake.create_all()

    Remove(app).run(reply, message, match)

    remove_user_from_group.assert_called_once_with('AuthorName', 'GroupName', 'MemberName')
    reply.append.assert_called_once_with('`MemberName` removed from Group `GroupName`, 0 total Members.')
