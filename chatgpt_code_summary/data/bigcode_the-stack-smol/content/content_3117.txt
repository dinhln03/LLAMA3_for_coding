
from pytest_bdd import given, when, then
from model.contact import Contact
import random


@given('a contact list')
def contact_list(db):
    return db.get_contact_list()

@given('a contact with <firstname> and <lastname>')
def new_contact(firstname, lastname):
    return Contact(firstname=firstname, lastname=lastname)

@when('I add the contact to the list')
def add_new_contact(app, new_contact):
    app.contact.create(new_contact)

@then('the new contact list is equal to the old contact list with the added contact')
def verify_contact_added(db, contact_list, new_contact):
    old_contacts = contact_list
    new_contacts = db.get_contact_list()
    old_contacts.append(new_contact)
    assert sorted(old_contacts, key=Contact.id_or_max) == sorted(new_contacts, key=Contact.id_or_max)

@given('a non-empty contact list')
def non_empty_contact_list(app, db):
    if len(db.get_contact_list()) == 0:
        app.contact.create(Contact(firstname='any firstname'))
    return db.get_contact_list()

@given('a random contact from the list')
def random_contact(non_empty_contact_list):
    return random.choice(non_empty_contact_list)

@when('I delete the contact from the list')
def delete_contact(app, random_contact):
    app.contact.delete_contact_by_id(random_contact.id)

@then('the new contact list is equal to the old contact list without the deleted contact')
def verify_contact_deleted(db, non_empty_contact_list, random_contact, app, check_ui):
    old_contacts = non_empty_contact_list
    new_contacts = db.get_contact_list()
    old_contacts.remove(random_contact)
    if check_ui:
        assert sorted(new_contacts, key=Contact.id_or_max) == sorted(app.contact.get_contact_list(), key=Contact.id_or_max)

@given('a non-empty contact list')
def non_empty_contact_list(app, db):
    if len(db.get_contact_list()) == 0:
        app.contact.create(Contact(firstname='any firstname'))
    return db.get_contact_list()

@when('I modify the contact from the list')
def modify_contact(app, random_contact):
    contact = Contact(firstname="New firstname")
    app.contact.modify_contact_by_id(random_contact.id, contact)

@then('the new contact list is equal to the old contact list without the deleted contact')
def verify_contact_deleted(db, non_empty_contact_list, random_contact, app, check_ui):
    old_contacts = non_empty_contact_list
    index = old_contacts.index(random_contact)
    new_contacts = db.get_contact_list()
    old_contacts[index] = random_contact
    if check_ui:
        assert sorted(new_contacts, key=Contact.id_or_max) == sorted(app.contact.get_contact_list(), key=Contact.id_or_max)
