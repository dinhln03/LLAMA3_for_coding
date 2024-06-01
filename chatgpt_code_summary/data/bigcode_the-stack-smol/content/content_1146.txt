import xlrd
import os
import sys
import copy
import json
import codecs
from collections import OrderedDict

# Constant Values
PARENT_NAME_ROW = 0
PARENT_NAME_COL = 0
COLUMN_NAMES_ROW = 1
DATA_STARTING_ROW = 2
ROOT_NAME = '*root'
ID_COLUMN_NAME = 'id'
PARENT_COLUMN_NAME = '*parent'
IGNORE_WILDCARD = '_'
REQUIRE_VERSION = (3, 5)
EXCEL_PATH = './excel/'
JSON_PATH = '../../asset/json/'


# Class
class TypeUtility:
    # xlrd is giving number as float

    @staticmethod
    def check_integer(value):
        return type(value) == float and int(value) == value

    # xlrd is giving boolean as integer
    @staticmethod
    def check_boolean(value):
        return type(value) == int

    @staticmethod
    def convert_value(value):
        if TypeUtility.check_integer(value):
            return int(value)
        elif TypeUtility.check_boolean(value):
            return bool(value)
        else:
            return value


class Table:

    def __init__(self, sheet):
        self.init_name(sheet)
        self.init_parent_name(sheet)
        self.init_metadata(sheet)
        self.init_descriptors(sheet)
        self.init_id_index_map()

    def init_name(self, sheet):
        self.name = sheet.name

    def init_parent_name(self, sheet):
        row = sheet.row_values(PARENT_NAME_ROW)
        self.parent_name = row[PARENT_NAME_COL]
        if type(self.parent_name) is not str:
            raise Exception('[' + self.name + ']' + 'Parent name is not string')
            sys.exit()

        self.is_root = self.parent_name == ROOT_NAME

    def init_metadata(self, sheet):
        row = sheet.row_values(COLUMN_NAMES_ROW)
        self.is_parent = False
        self.is_child = False
        self.column_names = []
        for value in row:
            if type(value) is not str:
                raise Exception('[' + self.name + ']' + 'Column name is not string')
                sys.exit()

            if value == ID_COLUMN_NAME:
                self.is_parent = True
            if value == PARENT_COLUMN_NAME:
                self.is_child = True
            self.column_names.append(value)

        if self.is_root and self.is_child:
            raise Exception('[' + self.name + ']' + 'Root table must not have a "' + PARENT_COLUMN_NAME + '" column')
            sys.exit()

        if not self.is_root and not self.is_child:
            raise Exception('[' + self.name + ']' + 'Child table must have a "' + PARENT_COLUMN_NAME + '" column')
            sys.exit()

    def init_descriptors(self, sheet):
        self.descriptors = []
        id_table = []
		
        for i in range(DATA_STARTING_ROW, sheet.nrows):
            #add metadata row count
            rowcount = i + 1
            col = sheet.row_values(i)
            desc = self.get_descriptor(col)

            if self.is_parent:
                id = desc[ID_COLUMN_NAME]
                
                if not id:
                    raise Exception('[' + self.name + ']' + 'Descriptor id must have a value - row : ' + str(i + 1))
                    sys.exit()
                
                if id in id_table:
                    raise Exception('[' + self.name + ']' + 'Descriptor id is duplicated - row : ' + str(i + 1))
                    sys.exit()

                id_table.append(id)
                
            self.descriptors.append(desc)

    def get_descriptor(self, col):
        descriptor = OrderedDict()
        for i in range(0, len(col)):
            key = self.column_names[i]
            if key[0] == IGNORE_WILDCARD:
                continue
            
            descriptor[key] = TypeUtility.convert_value(col[i])
            
        return descriptor

    def init_id_index_map(self):
        if not self.is_parent:
            return

        self.id_index_map = {}
        for descriptor in self.descriptors:
            id = descriptor[ID_COLUMN_NAME]
            self.id_index_map[id] = self.descriptors.index(descriptor)

    def merge_child_table(self, table):
        self.add_child_descriptor_list(table.name)
        for descriptor in table.descriptors:
            parent_id = descriptor[PARENT_COLUMN_NAME]
            parent_idx = self.id_index_map[parent_id]
            parent_descriptor = self.descriptors[parent_idx]
            parent_descriptor[table.name].append(descriptor)

    def add_child_descriptor_list(self, name):
        for descriptor in self.descriptors:
            descriptor[name] = []

    def remove_parent_column(self):
        for descriptor in self.descriptors:
            del descriptor[PARENT_COLUMN_NAME]

    def save_to_json(self, pretty_print, export_path):
        if pretty_print:
            string = json.dumps(self.descriptors, ensure_ascii=False, indent=4)
        else:
            string = json.dumps(self.descriptors, ensure_ascii=False)

        with codecs.open(export_path + self.name + '.json', 'w', 'utf-8') as f:
            f.write(string)


class Converter:

    def __init__(self, pretty_print, export_path):
        self.pretty_print = pretty_print
        self.export_path = export_path

    def convert(self, filename):
        print(filename + ' convert starting...')

        sheets = Converter.get_sheets(filename)
        root_table, tables = Converter.get_tables(sheets)
        Converter.post_process(tables)
        root_table.save_to_json(self.pretty_print, self.export_path)

        print(filename + ' convert is Done\n')

    @staticmethod
    def get_sheets(filename):
        path = os.path.abspath(filename)
        workbook = xlrd.open_workbook(path)
        return workbook.sheets()

    @staticmethod
    def get_tables(sheets):
        tables = {}
        root_tables = []

        for sheet in sheets:
            if sheet.name[0] == IGNORE_WILDCARD:
                continue

            table = Table(sheet)
            tables[table.name] = table
            if table.is_root:
                root_tables.append(table)

        if len(root_tables) == 1:
            return root_tables[0], tables
        else:
            raise Exception('Root table must be one')
            sys.exit()

    @staticmethod
    def post_process(tables):
        for name, table in tables.items():
            if table.is_root:
                continue

            parent_table = tables[table.parent_name]
            if not parent_table.is_parent:
                raise Exception('Parent table must have a id column')
                sys.exit()

            parent_table.merge_child_table(table)
            table.remove_parent_column()

# Script
current_version = sys.version_info
if current_version < REQUIRE_VERSION:
    raise Exception('[eeror]You Need Python 3.5 or later')
    sys.exit()

json_path = sys.argv[1] if len(sys.argv) > 1 else './'
converter = Converter(True, JSON_PATH + json_path)
    
for path, dirs, files in os.walk(EXCEL_PATH):
    for file in files:
        if file[0] is "~":
            continue
        
        if os.path.splitext(file)[1].lower() == '.xlsx':
            converter.convert(EXCEL_PATH + file)
