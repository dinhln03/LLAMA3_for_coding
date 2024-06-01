import argparse
import locale
import sys

from datetime import datetime

from model import *  
from sql import *
from common import *
from util import *


def parse_arguments():
    '''
        Parse input arguments. Passing the API key is defined as mandatory.
    '''
    parser = argparse.ArgumentParser(description='Incrementally exports JSON orders data into CSV format and optionally into a SQLite DB.')
    parser.add_argument('-k', '--key', type=str, required=True, help='API key to be used to perform the REST request to the backend.')
    parser.add_argument('-l', '--locale', type=str, required=False, help='Specify the locale: it_IT for italian. Otherwise machine default one.')
    parser.add_argument('-d', '--db', action='store_true', required=False, help='Instruct the tool to load a SQLite database up.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Define datastore base path to csv/ and db/ folders (csv/ and db/ folders should be already created).')
    parser.add_argument('-n', '--number', type=int, required=True, help='Define how many records each REST call should pull down.')
    parser.add_argument('-c', '--customer', type=int, required=False, help='Define whether the customer table should be updated contextually: it requires the number of cycles per page (max 50 records')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_arguments()
    if args.locale:
        locale.setlocale(locale.LC_ALL, args.locale)
    else:
        locale.setlocale(locale.LC_ALL, 'en_GB')
    datastore_path = args.path
    nr_records = args.number
    
    if not is_path_existent('%s/%s' % (datastore_path, 'csv')):
        sys.exit(1)
    if not is_path_existent('%s/%s' % (datastore_path, 'db')):
        sys.exit(1)
    
    # load or refresh the customer table for enrichment
    if args.customer:
        customers = load_customers_pages(args.key, args.customer)
        persist_customers_to_sqlite(customers, datastore_path)
    
    # looking up the customers for successive enrichment of orders
    lookup = lookup_customers(datastore_path)
    
    orders = load_orders_pages(args.key, nr_records, lookup)
    print('info: loaded %d order(s)...' % len(orders))
    print(orders[0])
    print('info: all records between FIRST and LAST\n')
    print(orders[-1])
    
    export_to_csv(orders, datastore_path)
    print('info: CSV export successul %d order(s)' % len(orders))

    if args.db:
        export_to_sqlite(orders, datastore_path)

if __name__ == "__main__":
    main()
