from model.contact import Contact
from model.group import Group
import random


def test_add_contact_in_group(app, db):
    if app.contact.count() == 0:
        app.contact.create_new(Contact(firstname="Contact for deletion", middlename="some middlename", lastname="some last name"))
    if len(app.group.get_group_list()) == 0:
        app.group.create(Group(name="Group for deletion"))
    group_id = app.group.get_random_group_id()
    contacts_in_group = app.contact.get_contacts_in_group(group_id)
    if len(contacts_in_group) > 0:
        contact = random.choice(contacts_in_group)
        app.contact.remove_from_group(contact.id, group_id)
        contact_ui = app.contact.get_contacts_in_group(group_id)
        contact_db = db.get_contacts_in_group(group_id)
        print()
        print(contact_db)
        print(contact_ui)
        assert contact_db == contact_ui
    else:
        True
    #
    # contact = app.contact.get_contacts_in_group(group_id)
    #
    # contacts = db.get_contact_list()
    #
    # contact = random.choice(contacts)
    # app.contact.add_contact_to_group(contact.id, group_id)
    #
    # contact_db = db.get_contacts_in_group(group_id)
    # assert contact_db == contact_ui