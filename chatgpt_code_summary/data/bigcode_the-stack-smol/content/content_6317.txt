import pymysql.cursors



from model.group import Group
from model.contact import Contact
class DbFixture:
    def __init__(self,host,name,user,password):
        self.host=host
        self.name=name
        self.user=user
        self.password=password
        self.connection=pymysql.connect(host=host ,database=name,user=user,password=password,autocommit=True)


    def get_group_list(self):
        list=[]
        cursor=self.connection.cursor()
        try:
            cursor.execute("select group_id,group_name,group_header,group_footer from group_list  where deprecated='0000-00-00 00:00:00'")
            for row in cursor:
                (id,name,header,footer)=row
                list.append(Group(id=str(id),name=name,header=header,footer=footer))
        finally:
            cursor.close()
        return list

    def get_contact_list(self):
        list=[]
        cursor=self.connection.cursor()
        try:
            cursor.execute("select id,firstname,lastname from addressbook  where deprecated='0000-00-00 00:00:00'")
            for row in cursor:
                (id,firstname,lastname)=row
                list.append(Contact(id=str(id),firstname=firstname,lastname=lastname))
        finally:
            cursor.close()
        return list


    def get_full_contact_list(self):
        list=[]
        cursor=self.connection.cursor()
        try:
            cursor.execute("select id,firstname,lastname,address, CONCAT (email  ,email2,email3), CONCAT (home,mobile ,work, phone2) from addressbook where deprecated='0000-00-00 00:00:00'")
            for row in cursor:
                (id,firstname,lastname,adress,fullemail,fullpfone)=row
                list.append(Contact(id=str(id),firstname=firstname,lastname=lastname,all_emails_from_home_page=fullemail,all_phones_from_home_page=fullpfone,address=adress))
        finally:
            cursor.close()
        return list



    def destroy(self):
        self.connection.close()