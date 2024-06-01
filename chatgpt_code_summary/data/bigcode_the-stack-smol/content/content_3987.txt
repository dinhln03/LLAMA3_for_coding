#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
"""
Manage database of SV JJ device fab/measurement parameters
File format: SQLite
Tables: barrier (dep structure), shape, josephson (measured params),
        trend (fitted Jc, RnA, IcRn)

BB, 2015
"""

print('hello')
import sqlite3

# Restrict table or column name for security
def scrub(table_name):
    return ''.join( chr for chr in table_name if chr.isalnum() or chr=='_' )

# Build string: 'column1=?,column2=?,...'
def assignform(collist):
    s = ' '
    for col in collist:
        s += col + '=?,'
    return s.rstrip(',')

class SVJJDB():
    def __init__(self, filename='svjj.db'):
        
        # Table structures
        self.colnames = {
                'barrier': ['wafer', 'chip', 'structure', 'fm1_name',
                    'fm1_thickness', 'fm2_name', 'fm2_thickness'],
                'shape': ['wafer', 'chip', 'device', 'shape', 'dim1', 'dim2'],
                'josephson': ['wafer', 'chip', 'device', 'temperature',
                    'ic_p', 'ic_ap', 'r_p', 'r_ap'],
                #'trend': ['wafer', 'structure', 'jc_p', 'jc_ap',
                #    'fm1_thickness', 'fm2_name', 'fm2_thickness'],
                }
        self.datatypes = {'wafer': 'string', 'chip':'string',
                'structure': 'string','shape':'string', 'fm1_name': 'string',
                'fm2_name': 'string', 'fm1_thickness': 'float',
                'fm2_thickness': 'float', 'device': 'string', 'dim1': 'float',
                'dim2': 'float', 'temperature': 'float', 'ic_p': 'float',
                'ic_ap': 'float', 'r_p': 'float', 'r_ap': 'float'}
        
        # Default values
        self.val0 = {'wafer': 'B150323a', 'chip': '56',
                'structure': 'Fe/Cu/Ni/Cu', 'shape':'ellipse', 'fm1_name': 'Fe',
                'fm2_name': 'Ni', 'fm1_thickness': '1e-9',
                'fm2_thickness': '2.4e-9', 'device': 'A01', 'dim1': '1e-6',
                'dim2': '1e-6', 'temperature': 4, 'ic_p': '10e-6',
                'ic_ap': '5e-6', 'r_p': '1', 'r_ap': '1'}

        self.conn = sqlite3.connect(filename)
        self.c = self.conn.cursor()

    def create_tables(self):
        
        # Create barrier structure table
        self.c.execute('''CREATE TABLE barrier
                (wafer text, chip text, structure text,
                fm1_name text, fm1_thickness real,
                fm2_name text, fm2_thickness real)''') 
        
        # Create device shape table
        self.c.execute('''CREATE TABLE shape
                (wafer text, chip text, device text,
                shape text, dim1 real, dim2 real)''') 

        # Create josephson measurement result table
        self.c.execute('''CREATE TABLE josephson
                (wafer text, chip text, device text, temperature real,
                ic_p real, ic_ap real, r_p real, r_ap real)''') 

    def close(self, save=True):
        if save: self.conn.commit()  # save
        self.conn.close()

    # Insert a row in barrier table
    def insert_row(self, table, arg):
        s1 = 'INSERT INTO %s VALUES ' % scrub(table)
        s2 = '(' + '?,'*(len(arg)-1) + '?)'
        self.c.execute(s1+s2, arg)
    
    def print_table(self, table):
        print(self.colnames[table])
        for row in self.c.execute('SELECT * FROM %s'%scrub(table)):
            print(row)
    #def print_table(self, table, ordercol):
        #for row in self.c.execute('SELECT * FROM %s ORDER BY ?'%scrub(table),\
        #        (ordercol,)):

    def delete_row(self, table, args):
        if table == 'barrier':
            self.c.execute('DELETE FROM %s WHERE wafer=? AND chip=?'
                    % scrub(table), args)
        elif table == 'shape' or table == 'josephson':
            self.c.execute('DELETE FROM %s WHERE '
                    'wafer=? AND chip=? AND device=?' % scrub(table), args)
        else:
            print('No table name: %s' % table)

    def update_row(self, table, vals, **newassign):

        s1 = 'UPDATE %s' % scrub(table)
        s2 = ' SET' + assignform(self.colnames[table])
        s3 = ' WHERE' + assignform(matchcols)
        print(s1+s2+s3)
        #self.c.execute(s1 + s2 + s3, vals + matchvals)

# Derived class for interactive shell execution
class SVJJDBInteract(SVJJDB):
    #def create_db(self, *arg):
    #    self.create_tables(*arg)  # pass filename

    def print(self, table):
        self.print_table(table)

    # Get inputs from argument or interactively
    # Use val0 as default for interactive case
    def input_param(self, key, val0='0', **kwargs):
        interact = kwargs.get('interact', True)
        datatype = kwargs.get('datatype', 'string')
        if interact:
            msg = input(key + '? [%s] '%str(val0))
            if msg == '': msg = val0          # empty input means default

            if datatype == 'string': val = msg
            if datatype == 'int': val = int(msg)
            if datatype == 'float': val = float(msg)
        else:
            val = val0
        return val

    def insert(self, table):
        vals = ()
        for col in self.colnames[table]:
            vals = vals + (self.input_param(col, self.val0[col],\
                    datatype=self.datatypes[col], interact=True),)

        self.insert_row(table, vals)

    def delete(self, table, *args):
        self.delete_row(table, args)

    # *args = wafer, chip, [device]
    def update(self, table, **newassign):            
        # Load previous values as default (val0)
        
        # User input
        vals = ()
        for col in self.colnames[table]:
            vals = vals + (self.input_param(col, self.val0[col],\
                    datatype=self.datatypes[col], interact=True),)

        self.update_row(table, vals, **newassign)
        
    # Pass on any SQL statement
    def execsql(self, *cmd):
        self.c.execute(cmd[0])

def args2kwargs(args):
    l = []
    for arg in args:
        l += [arg.split('=')]
    return dict(l)

# main shell interface (run SVJJDBInteract class)
def app(argv):
    """Execute in system shell
    """
    if len(argv) < 2:
        print("Usage: python %s <command> <table> [<column1>=<value1> [...]]\n" 
              "       <command>: print, insert, delete, or edit\n"
              "       <table>: barrier, shape, or josephson\n" % argv[0])
        sys.exit(0)

    db = SVJJDBInteract()

    # Fixed arguments
    funcname = argv[1]
    table = argv[2]

    # Convert extra to keyword arguments
    kwargs = args2kwargs(argv[3:])
    getattr(db, funcname)(table, **kwargs)
    db.close()

# simple test run
def app2(argv):
    db = SVJJDB()
    #db.create_tables()
    db.insert_row('barrier', ('B150413', '22', 'Fe/Cu/Ni/Cu', 'Fe', 1e-9,\
            'Ni', 2.4e-9))
    db.print_table('barrier', 'chip')
    db.close()

if __name__ == '__main__':
    import sys
    print(sys.version)
   
    app(sys.argv)
    print('Bye!')
