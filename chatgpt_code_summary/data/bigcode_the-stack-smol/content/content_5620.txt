from collections import namedtuple
from datetime import datetime
import pprint
import pickle
import json

import sqlite3

BUILDINGS = ['AGNEW', 'LARNA', 'AJ E', 'AJPAV', 'AQUAC', 'AA', 'ADRF', 'ARMRY', 'ART C', 'BEEF', 'BFH', 'BRNCH', 'PRC', 'BURCH', 'BUR', 'CAM', 'CARNA', 'CAM M', 'CAPRI', 'CSB', 'LIBR', 'COL', 'CMMID', 'CSSER', 'CHEMP', 'COLSQ', 'COLS2', 'CEC', 'CO', 'CIC', 'DAIRY', 'DAV', 'DER', 'DTRIK', 'DBHCC', 'DB', 'DTNA', 'DURHM', 'EGG E', 'ENGEL', 'FEM', 'FST', 'FRALN', 'FSBRN', 'GBJ', 'GOODW', 'GLCDB', 'GH', 'HAHN N', 'HAHN S', 'HARP', 'HEND', 'HILL', 'HOLD', 'HORSE', 'HABB1', 'HUTCH', 'CRCIA', 'TC', 'ICTAS', 'ILSB', 'HAN', 'JCH', 'KENT', 'KAMF', 'LANE', 'STAD', 'LATH', 'WLH', 'LEE', 'LARTS', 'LFSCI', 'LITRV', 'LYRIC', 'MAJWM', 'EMPOR', 'MCB', 'MCCOM', 'MRYMN', 'MIL', 'MAC', 'NCB', 'NEB', 'NHW', 'NEW', 'NOR', 'GRNDP', 'SEC', 'OWENS', 'PK', 'PAM', 'PAT', 'PAYNE', 'PY', 'PAB', 'PWCOM', 'PRICE', 'P HSE', 'PRINT', 'PRT E', 'PSC', 'RAND', 'RCTR1', 'ROB', 'SANDY', 'SARDO', 'SAUND', 'SEITZ', 'SHANK', 'SHEEP', 'SHULT', 'SEB', 'SL TW', 'SLUSH', 'SM CC', 'SMYTH', 'SOL', 'SQUIR', 'SURGE', 'SWINE', 'HAHN', 'TESKE', 'T101', 'CPAP', 'TORG', 'UPEST', 'VTC', 'NOVAC', 'NVC', 'VMIA', 'VM 1', 'VM 2', 'VM 3', 'VM4C1', 'WAL', 'GYM', 'WHIT', 'WMS', 'BROOK', 'BFPC']
BUILDING_NAMES = ['Agnew Hall', 'Alphin-Stuart Arena', 'Ambler Johnston East', 'Animal Judging Pavilion', 'Aquaculture Facility', 'Architectural Annex', 'Architecture Demo & Res Fac', 'Armory (Art)', 'Art & Design Learning Center', 'Beef Barn', 'Bishop-Favrao Hall', 'Branch Building', 'Brooder House', 'Burchard Hall', 'Burruss Hall', 'Campbell Arena', 'Campbell Arena', 'Campbell Main', 'Capri Building', 'Career Services Building', 'Carol M. Newman Library', 'Cassell Coliseum', 'Center Molecular Med Infec Dis', 'Center for Space Sci & Engr Re', 'Chemistry Physics Building', 'Collegiate Square', 'Collegiate Square Two', 'Continuing Education Center', 'Cowgill Hall', 'Cranwell Int\'l Center', 'Dairy Barn', 'Davidson Hall', 'Derring Hall', 'Dietrick Hall', 'Donaldson Brown Hotel & Conf', 'Donaldson-Brown Hall', 'Downtown North', 'Durham Hall', 'Eggleston East', 'Engel Hall', 'Femoyer Hall', 'Food Science & Technology Lab', 'Fralin Biotechnology Center', 'Free Stall Barn', 'G. Burke Johnston Student Ctr', 'Goodwin Hall', 'Graduate Life Ctr Dnldsn Brown', 'Greenhouse', 'Hahn Hall North Wing', 'Hahn Hall South Wing', 'Harper', 'Henderson Hall', 'Hillcrest', 'Holden Hall', 'Horse Barn', 'Human & Ag Biosciences Bldg 1', 'Hutcheson Hall', 'ICTAS A', 'Indoor Tennis Courts', 'Inst for Crit Tech & Appld Sci', 'Integrated Life Sciences Bldg', 'John W. Hancock Jr. Hall', 'Julian Cheatham Hall', 'Kent Square', 'Kroehling Adv Material Foundry', 'Lane Hall', 'Lane Stadium', 'Latham Hall', 'Lavery Hall', 'Lee', 'Liberal Arts Building', 'Life Sciences I', 'Litton-Reaves Hall', 'Lyric Theater', 'Major Williams Hall', 'Math Emporium', 'McBryde Hall', 'McComas Hall', 'Merryman Athletic Facility', 'Military Building/Laundry', 'Moss Arts Center', 'New Classroom Building', 'New Engineering Building', 'New Hall West', 'Newman', 'Norris Hall', 'Old Grand Piano Building', 'Old Security Building', 'Owens Hall', 'PK', 'Pamplin Hall', 'Patton Hall', 'Payne', 'Peddrew-Yates', 'Performing Arts Building', 'Pointe West Commons', 'Price Hall', 'Price House', 'Print Shop', 'Pritchard East', 'Psychological Services Center', 'Randolph Hall', 'Riverside Center 1', 'Robeson Hall', 'Sandy Hall', 'Sardo Laboratory', 'Saunders Hall', 'Seitz Hall', 'Shanks Hall', 'Sheep Barn', 'Shultz Hall', 'Signature Engineering Building', 'Slusher Tower', 'Slusher Wing', 'Smith Career Center', 'Smyth Hall', 'Solitude', 'Squires Student Center', 'Surge Space Building', 'Swine Center', 'T. Marshall Hahn, Jr. Hall', 'Teske House', 'Theatre 101', 'Thomas Conner House', 'Torgersen Hall', 'Urban Pest Control Facility', 'VT/Carilion Medicl Sch/Res Ins', 'VT/UVA Northern VA Ctr', 'VT/UVA Northern VA Ctr', 'Vet Med Instructional Addition', 'Vet Med Phase 1', 'Vet Med Phase 2', 'Vet Med Phase 3', 'Vet Med Phase 4C-Non-Client', 'Wallace Hall', 'War Memorial Gymnasium', 'Whittemore Hall', 'Williams Hall', 'Wood Engineering Lab', 'Wood Processing Lab']

TERMS = ['201601', '201606', '201607', '201609', '201612', '201701', '201706', '201707', '201709', '201712', '201801', '201806', '201807', '201809', '201812', '201901', '201906', '201907', '201909', '201912', '202001', '202006', '202007', '202009', '202012']
TERM_NAMES = ['Spring 2016', 'Summer I 2016', 'Summer II 2016', 'Fall 2016', 'Winter 2017', 'Spring 2017', 'Summer I 2017', 'Summer II 2017', 'Fall 2017', 'Winter 2018', 'Spring 2018', 'Summer I 2018', 'Summer II 2018', 'Fall 2018', 'Winter 2019', 'Spring 2019', 'Summer I 2019', 'Summer II 2019', 'Fall 2019', 'Winter 2020', 'Spring 2020', 'Summer I 2020', 'Summer II 2020', 'Fall 2020 (tentative)', 'Winter 2021']

DAYS = ['U', 'M', 'T', 'W', 'R', 'F', 'S']

ATTRIBS = ['06', 'AV01', 'AV02', 'AV03', 'AV04', 'AV05', 'AV06', 'AV07', 'AV08', 'AV09', 'AV10', 'AV11', 'AV12', 'AV15', 'FX13', 'FX14', 'PC04', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC30', 'PC31', 'PC32', 'ST20', 'ST21', 'ST22', 'ST23', 'ST24', 'ST29', 'TY20', 'TY24', 'TY25', 'TY26', 'TY27', 'TY28', 'TY32',  'TY33']
ATTRIB_NAMES = ['DO NOT USE WHD', 'Sound System', 'Wireless Microphone', 'Slide Projector', 'Technical Support Needed', 'WiFi', 'DVD/VCR Combination Unit', 'Computer Projection', 'Document Camera', 'DVD/Blueray Player', 'Crestron Control', 'AC Outlets at seats/floor', 'Computer Installed', 'Lecture Capture', 'Chalk Board - Multiple', 'Whiteboard - Multiple', 'Do not use', 'Windows', 'Shades or Blinds', 'Handicapped Accessible', 'Do not use', 'Air Conditioning', 'General Assignment Classroom', 'Non General Assignment Clsrm', 'Gen Assignment Restricted Use', 'Tablet Arm Chairs', 'Fixed Seating', 'Tables (round)', 'Do not use', 'Sled style desk', 'Moveable Chairs and Tables', 'Do not use', 'Do not use', 'Computer Lab', 'Auditorium', 'Scale up', 'Interactive Classroom', 'Classroom', 'Tiered Classroom']

ClassBlock = namedtuple('ClassBlock', ['room', 'name', 'crn'])
EventBlock = namedtuple('EventBlock', ['room', 'times', 'name'])

RoomAttribs = namedtuple('RoomAttribs', ['capacity', 'attribs'])

with open('201801-rooms.dat', 'rb') as f:
    rooms = pickle.load(f)

with open('201801-blocks.dat', 'rb') as f:
    blocks = pickle.load(f)

with open('201801-attribs.dat', 'rb') as f:
    attribs = pickle.load(f)



conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

c = conn.cursor()

c.execute('''
CREATE TABLE buildings
(code TEXT PRIMARY KEY, name TEXT, latitude DECIMAL(9,6), longitude DECIMAL(9,6))
''')

c.execute('''
CREATE TABLE rooms
(id INT PRIMARY KEY, building_code TEXT, room TEXT, capacity INT, attribs TEXT,
    FOREIGN KEY(building_code) REFERENCES buildings(code))
''')

c.execute('''
CREATE TABLE classes
(id INT PRIMARY KEY, room_id INT, days TEXT, start TIMESTAMP, end TIMESTAMP, crn TEXT, name TEXT,
    FOREIGN KEY(room_id) REFERENCES rooms(id))
''')

c.execute('''
CREATE TABLE events
(id INT PRIMARY KEY, room_id INT, days TEXT, start_day TIMESTAMP, end_day TIMESTAMP, start TIMESTAMP, end TIMESTAMP, name TEXT,
    FOREIGN KEY(room_id) REFERENCES rooms(id))
''')

conn.commit()

# Load buildings and rooms
for building, rooms in attribs.items():
    c.execute('''
INSERT INTO buildings
(code, name) values (?, ?)
    ''', (building, BUILDING_NAMES[BUILDINGS.index(building)]))
    
    for room, value, in rooms.items():
        c.execute('''
INSERT INTO rooms
(building_code, room, capacity, attribs) values (?, ?, ?, ?)
        ''', (building, room, value[0], ','.join(value[1])))

# Load classes
