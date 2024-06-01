import sqlite3
import datetime
from collections import Counter
import calendar

def return_unique_ID():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM rolodex')
    IDs = []
    for item in c:
        ID = int(item[0])
        IDs.append(ID)
    IDs = sorted(IDs, key=int, reverse=True)
    uniqueID = IDs[0] + 1
    return str(uniqueID)

def return_unique_order_ID():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM orders')
    IDs = []
    for item in c:
        ID = int(item[0])
        IDs.append(ID)
    IDs = sorted(IDs, key=int, reverse=True)
    uniqueID = IDs[0] + 1
    return str(uniqueID)


def input_entry(customerFirstName, customerLastName, customerPhoneNumber, customerAddress, customerPayMethod):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    uniqueID = return_unique_ID()
    rolodexEntry = (uniqueID, customerFirstName, customerLastName, customerPhoneNumber, customerAddress, customerPayMethod)
    c.execute('INSERT INTO rolodex VALUES (?,?,?,?,?,?)', rolodexEntry)
    conn.commit()


def return_all_entries():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM rolodex ORDER BY last_name')
    return c


def review_all_entries():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM rolodex')
    for item in c:
        print(item[0], item[1], item[2], item[3], item[4])


def delete_entry_by_id(uniqueID):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('DELETE FROM rolodex WHERE id = ?', uniqueID)
    conn.commit()


def update_rolodex_entry(variable, variable_type, uniqueID):
    if variable_type == "name":
        update_rolodex_entry_name(variable, uniqueID)
    elif variable_type == "phoneNumber":
        update_rolodex_entry_phoneNumber(variable, uniqueID)
    elif variable_type == "address":
        update_rolodex_entry_address(variable, uniqueID)
    elif variable_type == "payMethod":
        update_rolodex_entry_payMethod(variable, uniqueID)
    else:
        print("failed to update anything.")


def update_rolodex_entry_name(variable, uniqueID):
    combo = (variable, uniqueID)
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('''UPDATE rolodex
                    SET name = ?
                    WHERE id = ?''', combo)
    conn.commit()


def update_rolodex_entry_phoneNumber(variable, uniqueID):
    combo = (variable, uniqueID)
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('''UPDATE rolodex
                    SET phoneNumber = ?
                    WHERE id = ?''', combo)
    conn.commit()


def update_rolodex_entry_address(variable, uniqueID):
    combo = (variable, uniqueID)
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('''UPDATE rolodex
                    SET address = ?
                    WHERE id = ?''', combo)
    conn.commit()


def update_rolodex_entry_payMethod(variable, uniqueID):
    combo = (variable, uniqueID)
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('''UPDATE rolodex
                    SET payMethod = ?
                    WHERE id = ?''', combo)
    conn.commit()


def create_rolodex_table():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    create_table = """CREATE TABLE IF NOT EXISTS rolodex (
                      id integer PRIMARY KEY,
                      first_name text,
                      last_name text,
                      phone_number int,
                      address text,
                      payMethod text)
                      """
    c.execute(create_table)
    conn.commit()


def search_by_customer_id(customer_id):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    customer_id = (customer_id,)
    c.execute('''SELECT * FROM rolodex WHERE id = (?)''', customer_id)
    return c


def search_by_customer_first_name(customer_name):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    customer_name = (customer_name,)
    c.execute('''SELECT * FROM rolodex WHERE first_name = (?)''', customer_name)
    return c

def search_by_customer_last_name(customer_name):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    customer_name = (customer_name,)
    c.execute('''SELECT * FROM rolodex WHERE last_name = (?)''', customer_name)
    return c

def search_by_customer_phone_number(customer_phone_number):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    customer_phone_number = (customer_phone_number,)
    c.execute('''SELECT * FROM rolodex WHERE phone_number = (?)''', customer_phone_number)
    return c


def create_orders_table():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    create_table = """CREATE TABLE orders (
                      id integer PRIMARY KEY,
                      custid SMALLINT,
                      orderlist text,
                      time_stamp text)
                      """
    c.execute(create_table)
    conn.commit()


def create_customerprefs_table():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    create_table = """CREATE TABLE IF NOT EXISTS customerprefs (
                      id integer PRIMARY KEY,
                      customer_id integer,
                      field_id integer,
                      description text)
                      """
    c.execute(create_table)
    conn.commit()


def new_customer_delivery_preference(customerID, customer_delivery_preference):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    uniqueID = return_unique_customerpreference_ID()
    orderEntry = (uniqueID, customerID, 10, customer_delivery_preference)
    c.execute('INSERT INTO customerprefs VALUES (?,?,?,?)', orderEntry)
    conn.commit()

def return_unique_customerpreference_ID():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM customerprefs')
    IDs = []
    for item in c:
        ID = int(item[0])
        IDs.append(ID)
    IDs = sorted(IDs, key=int, reverse=True)
    uniqueID = IDs[0] + 1
    return str(uniqueID)


def input_new_order(customerID, order_list):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    uniqueID = return_unique_order_ID()
    orderEntry = (uniqueID, 1, order_list, datetime.datetime.now())
    c.execute('INSERT INTO orders VALUES (?,?,?,?)', orderEntry)
    conn.commit()


#def drop_rolodex_table():
#    conn = sqlite3.connect("ORDERM8.db")
#    c = conn.cursor()
#    c.execute('DROP table rolodex')
#    for item in c:
#        orderlist = item[2].split()
#        print item[0], item[1], orderlist, item[3]

def return_all_customerprefs_entries():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM customerprefs')
    return c


def get_latest_customerprefs(customer_id):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    customer_id_tuple = (customer_id,)
    c.execute('SELECT * FROM customerprefs WHERE customer_id=(?) AND field_id = 10 ORDER BY id DESC LIMIT 1',
              customer_id_tuple)
    for item in c:
        return item


def get_latest_foodprefs(customer_id):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    customer_id_tuple = (customer_id,)
    c.execute('SELECT * FROM customerprefs WHERE customer_id=(?) AND field_id = 20 ORDER BY id DESC LIMIT 1',
              customer_id_tuple)
    for item in c:
        return item


def new_customer_food_preference(customerID, customer_food_preference):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    uniqueID = return_unique_customerpreference_ID()
    orderEntry = (uniqueID, customerID, 20, customer_food_preference)
    c.execute('INSERT INTO customerprefs VALUES (?,?,?,?)', orderEntry)
    conn.commit()


def delete_customer_and_customer_records(customerID):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    id = (customerID,)
    c.execute('DELETE FROM rolodex WHERE id=(?)', id)
    c.execute('DELETE FROM customerprefs WHERE customer_id=(?)', id)
    conn.commit()


# Day Duties Stuff.


def create_day_duties_table():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    create_table = """CREATE TABLE IF NOT EXISTS day_duties (
                      id integer PRIMARY KEY,
                      date_of_entry DATE,
                      day_of_week text,
                      task text)
                      """
    c.execute(create_table)
    conn.commit()


def return_unique_day_duty_ID():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM day_duties')
    IDs = []
    for item in c:
        ID = int(item[0])
        IDs.append(ID)
    IDs = sorted(IDs, key=int, reverse=True)
    uniqueID = IDs[0] + 1
    return str(uniqueID)


def new_day_duty(date_of_entry, day_of_week, task):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    uniqueID = return_unique_day_duty_ID()
    dutyEntry = (uniqueID, date_of_entry, day_of_week, task)
    c.execute('INSERT INTO day_duties VALUES (?,?,?,?)', dutyEntry)
    conn.commit()


def return_all_day_duties():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM day_duties')
    return c


def search_by_day_of_week(day_of_week):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    day_of_week = (day_of_week,)
    c.execute('''SELECT * FROM day_duties WHERE day_of_week = (?)''', day_of_week)
    return c

# Examples

# new_day_duty(datetime.datetime.now(), "Wednesday", "Condense Recycling")

# for item in return_all_day_duties():
#     print item


# DAILY CUSTOMER ENTRIES

def delete_daily_customer_entrys(order_id):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    order_id = (str(order_id),)
    c.execute('''DELETE FROM daily_customers WHERE id=(?)''',order_id)
    conn.commit()


def return_unique_daily_customer_entry_id():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM daily_customers')
    IDs = []
    for item in c:
        ID = int(item[0])
        IDs.append(ID)
    IDs = sorted(IDs, key=int, reverse=True)
    uniqueID = IDs[0] + 1
    return str(uniqueID)


def create_daily_customers_table():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    create_table = """CREATE TABLE daily_customers (
                      id integer PRIMARY KEY,
                      custid SMALLINT,
                      todays_date DATE)
                      """
    c.execute(create_table)
    conn.commit()


def new_daily_customer(customer_id):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    uniqueID = return_unique_daily_customer_entry_id()
    dutyEntry = (uniqueID, customer_id, datetime.date.today(),)
    c.execute('INSERT INTO daily_customers VALUES (?,?,?)', dutyEntry)
    conn.commit()


def return_all_daily_customer_entries():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    date = (datetime.date.today(),)
    c.execute('SELECT * FROM daily_customers WHERE todays_date=(?)', date)
    return c


def weekly_graph_data():
    entries = return_this_weeks_customer_entries()
    daycount = enumerate_this_weeks_customer_entries(entries)
    week_dictionary = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for key, value in daycount.items():
        if key in week_dictionary:
            week_dictionary[key] = value
        else:
            pass
    return week_dictionary


def monthly_graph_data():
    entries = return_this_months_customer_entries()[0]
    start_end = return_this_months_customer_entries()[1]
    daycount = enumerate_this_months_customer_entries(entries)
    month_dictionary = {}
    dictionary_range = range(start_end[0].day,start_end[1].day+1,1)
    for item in dictionary_range:
        month_dictionary[item] = 0
    for key, value in daycount.items():
        if key in month_dictionary:
            month_dictionary[key] = value
        else:
            pass
    return month_dictionary


def yearly_graph_data():
    entries = return_this_years_customer_entries()
    daycount = enumerate_this_years_customer_entries(entries)
    week_dictionary = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
    for key, value in daycount.items():
        if key in week_dictionary:
            week_dictionary[key] = value
        else:
            pass
    return week_dictionary


def return_all_customer_entries_from_daily_customers():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    c.execute('SELECT * FROM daily_customers')
    return c

def return_this_weeks_customer_entries():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    today_date = (datetime.date.today(),)
    week_start = today_date[0] - datetime.timedelta(days=today_date[0].weekday())
    week_end = week_start + datetime.timedelta(days=4)
    week_tuple = (week_start, week_end,)
    c.execute('SELECT * FROM daily_customers WHERE todays_date BETWEEN (?) and (?)', week_tuple)
    return c


def enumerate_this_weeks_customer_entries(customer_entries):
    dates = []
    for item in customer_entries:
            dates.append(item[2])
    days = []
    for item in dates:
        days.append(datetime.datetime.strptime(item, "%Y-%m-%d").weekday())
    return Counter(days)


def return_this_months_customer_entries():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    today_date = datetime.date.today()
    start_of_month = datetime.datetime(today_date.year, today_date.month, 1)
    end_of_month = datetime.datetime(today_date.year,
                                     today_date.month,
                                     calendar.monthrange(today_date.year, today_date.month)[1])
    start_end_tuple = (start_of_month, end_of_month)
    c.execute('SELECT * FROM daily_customers WHERE todays_date BETWEEN (?) and (?)', start_end_tuple)
    return c, start_end_tuple


def enumerate_this_months_customer_entries(customer_entries):
    dates = []
    for item in customer_entries:
            dates.append(item[2])
    days = []
    for item in dates:
        days.append(datetime.datetime.strptime(item, "%Y-%m-%d").day)
    return Counter(days)


def return_this_years_customer_entries():
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    today_date = datetime.date.today()
    start_of_year = datetime.datetime(today_date.year, 1, 1)
    end_of_year = datetime.datetime(today_date.year, 12, 31)
    start_end_tuple = (start_of_year, end_of_year)
    c.execute('SELECT * FROM daily_customers WHERE todays_date BETWEEN (?) and (?)', start_end_tuple)
    return c, start_end_tuple


def enumerate_this_years_customer_entries(customer_entries):
    dates = []
    for item in customer_entries[0]:
            dates.append(item[2])
    days = []
    for item in dates:
        days.append(datetime.datetime.strptime(item, "%Y-%m-%d").month)
    return Counter(days)

# for messing around with daily customer entries

# for item in return_all_daily_customer_entries():
#    print item

# for item in range(0,15):
#    delete_daily_customer_entrys(item)


# FOR COPYING ROLODEX AND CUSTOMERPREFS FROM PEPS DB TO NEW DB
# will have to delete all entries on the new db in customerprefs and rolodex for this to work.


def create_test_empty_db():
    conn = sqlite3.connect("ORDERM8_test.db")
    conn.close()


def copy_table_db_to_db():
    conn = sqlite3.connect("ORDERM8_test.db")#newdb
    c = conn.cursor()
    c.execute("ATTACH 'ORDERM8.db' AS test")#pepsdb
    c.execute("INSERT INTO rolodex SELECT * FROM test.rolodex")
    conn.commit()

# create_test_empty_db()
# create_rolodex_table()

# will only need this function once rolodex and customerprefs are cleared out.
# copy_table_db_to_db()


def select_all_activity(customer_id):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    customer_id_tuple = (customer_id,)
    c.execute("SELECT * FROM daily_customers WHERE custid=(?)", customer_id_tuple)
    return c


def select_recent_activity(customer_id):
    conn = sqlite3.connect("ORDERM8.db")
    c = conn.cursor()
    customer_id_tuple = (customer_id,)
    c.execute("SELECT * FROM daily_customers WHERE custid=(?) ORDER BY todays_date DESC LIMIT 10", customer_id_tuple)
    return c
