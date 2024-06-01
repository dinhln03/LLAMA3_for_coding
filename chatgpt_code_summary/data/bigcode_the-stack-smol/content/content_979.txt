from model.contact import Contact
import random

def test_delete_some_contact(app, db, check_ui):
    if app.contacts.count() == 0:
        app.contacts.create_new_contact(Contact(firstname="crab"))
    old_contacts = db.get_contact_list()
    contact = random.choice(old_contacts)
    app.contacts.delete_contact_by_id(contact.id)
    new_contacts = db.get_contact_list()
    assert len(old_contacts) - 1 == app.contacts.count()
    old_contacts.remove(contact)
    assert old_contacts == new_contacts
    if check_ui:
        assert sorted(new_contacts, key=Contact.id_or_max) == sorted(app.contacts.get_contacts_list(), key=Contact.id_or_max)


