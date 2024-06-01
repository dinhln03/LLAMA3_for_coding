import random
import string
import os.path
import jsonpickle
import getopt
import sys
from model.contact import Contact

try:
    opts, args = getopt.getopt(sys.argv[1:], "n:f:", ["namber of group", "file"])
except getopt.GetoptError as err:
    getopt.usage()
    sys.exit(2)

n = 5
f = "/data/contacts.json"

for o, a in opts:
    if o == "-n":
        n = int(a)
    elif o == "-f":
        f = a


def random_string(prefix, maxlen):
    symbols = string.ascii_letters+string.digits+" "*10
    return prefix+"".join([random.choice(symbols) for i in range(random.randrange(maxlen))])

testdata = [Contact(firstname="", middlename="", lastname="", nickname="", title="", company="",
                    address="", homephone="", mobilephone="", workphone="", fax="", address2="",
                    seconderyphone="", notes="")]+ [Contact(firstname=random_string('firstname', 10),
                                                    middlename=random_string('middlename', 10),
                                                    lastname=random_string('lastname', 10),
                                                    nickname=random_string('nickname', 10),
                                                    title=random_string('title', 10),
                                                    company=random_string('company', 10),
                                                    address=random_string('address', 10),
                                                    homephone=random_string('home', 10),
                                                    mobilephone=random_string('mobile', 10),
                                                    workphone=random_string('work', 10),
                                                    fax=random_string('fax', 10),
                                                    address2=random_string('address2', 10),
                                                    seconderyphone=random_string('phone2', 10),
                                                    notes=random_string('notes', 10)) for i in range(5)
]

file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."+f)
with open(file, "w") as out:
    jsonpickle.set_encoder_options("json", indent=2)
    out.write(jsonpickle.encode(testdata))
