from model.contact import Contact
from random import randrange
def test_add_contact(app,db,check_ui):
    if len(db.get_contact_list()) == 0:
        app.contact.create(Contact(firstName='test'))
    old_contact = db.get_contact_list()
    index = randrange(len(old_contact))
    contact = Contact(firstName="firstName", middleName="middleName", lastName="lastName", nickName="nickName", title="title",
                               company="company", address="address", home="home", mobile="mobile", work="work", fax="fax", email="email",
                               email2="email2", email3="email3", homepage="homepage", address2="address2", phone2="phone2", notes="notes")
    contact.id=old_contact[index].id
    app.contact.edit_contact_by_id(contact.id, contact)
    new_contact = db.get_contact_list()
    assert len(old_contact) == len(new_contact)
    old_contact[index]= contact
    assert sorted(old_contact, key=Contact.id_or_map) == sorted(new_contact, key=Contact.id_or_map)
    if check_ui:
        assert sorted(new_contact, key=Contact.id_or_map) == sorted(app.contact.get_contact_list(), key=Contact.id_or_map)