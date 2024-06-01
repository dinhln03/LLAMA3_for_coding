import sqlite3;
import csv;
import sys;
from ordery.db import get_db
from flask import current_app
def order_csv(filename):
    ## Connect to the database
    try:
        conn = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES);            # Get a connection object for the database
        conn.execute('PRAGMA foreign_keys = ON;');  # Turn on foreign key constraints
        csr = conn.cursor();                        # Get a cursor object for the connection
    except Exception as e:
        print("Error connecting to database: ", e);  # Print error message
        sys.exit(); # Fatal Error
## Open the orders csv file
    try:
        f = open(filename, newline='');     # Open the file – default for reading
        r = csv.DictReader(f);                  # Return a dictionary reader iterator for the file
        print("\n csv file openned successfully")
    except Exception as e:
        print("Error opening csv file: ", e);   # Print error message
        sys.exit(); # Fatal Error


    ## --------------------------------------
    ## Loop through the orders csv file and insert each row in the table
    ## File title line: ord_nbr, prod_nbr, ord_qty, ord_date
    for d_row in r: # Loop on each row in the file into a list
        t_row = (int(d_row['ord_nbr']), d_row['prod_nbr'], int(d_row['ord_qty']), d_row['ord_date']);

        csr.execute('BEGIN TRANSACTION'); # Start transaction
        try:
            # Check if order number already exists
            v_sql = 'SELECT id FROM orders WHERE ord_nbr = ?';
            csr.execute(v_sql,(t_row[0],) );
            t_id = csr.fetchone();  # Get the order id
            if t_id != None:        # Order number already exists in orderss
                print("\nOrder number " + str(t[0]) + " already exists in orders table");
                continue;           # Get next order

            # Get product number id IF it exists in product table
            v_sql = 'SELECT id FROM products WHERE prod_nbr = ?';
            csr.execute(v_sql,(t_row[1],) );
            t_pid = csr.fetchone(); # Get the product id
            if t_pid == None:
                print("\nProduct number " + str(t_row[1]) + " does not exist in products table");
                continue;           # Get next order

            # If order number Not Exist and product number Exist then Insert the order
            if t_id == None and t_pid != None:
                v_sql = '''INSERT INTO orders (ord_nbr, ord_qty, ord_date, prod_id)
                                       VALUES (?,?,?,?);'''
                csr.execute(v_sql, (t_row[0], t_row[2], t_row[3], t_pid[0]) );
                conn.commit();      # Commit transaction for this row
        except Exception as e:
            print("Error loading Orders table " + str(e));  # Print error message
            print("Order number: ", t_row[0]);              # Identify order number
            conn.rollback();        # Rollback this transaction

    f.close();      # Close the file
    conn.close()
