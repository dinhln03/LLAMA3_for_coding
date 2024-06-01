"""
    The code below crawls the annotations of the MADE 1.0 Train Data and stores them
    as Corpus ID, Annotation ID, Type, Length, Offset, Text in the 
    CSV_Annotations.csv file.
    Input Files:
        All xml files in the annotations folder in the made_train_data folder
    Output Files:
        CSV_Annotations.csv
    Note: Make sure to delete the CSV_Annotations.csv file if already existing in 
    the folder as this code appends to the existing file.
"""

# Importing required Files

import os
import xml.etree.ElementTree as ET
import csv

final =list()
final.append(["Content ID", "Annotation ID", "Type", "Length", "Offset", "Text"])

# Reading required files 

path ="C:\\Project_NLP_Final\\Project Dataset\\made_train_data\\annotations\\"
dirListing = os.listdir(path)
for item in dirListing:
    tree = ET.parse(path + '\\' + item)
    root = tree.getroot() 
    annot = dict()
    for i in root.findall('./document/passage'):
        flag = 0
        for doc in i.findall('./annotation'):
            annot=list()  
            annot.append(item[0:-9])
            annot.append(doc.get('id'))
            for typ in doc:
                if typ.tag =='infon':
                    annot.append(typ.text)
                elif typ.tag =='location':
                    annot.append(typ.get('length'))
                    annot.append(typ.get('offset'))
                elif typ.tag == 'text':
                    annot.append(typ.text)
            final.append(annot)
            flag = 1
        if flag == 0:
            annot = [item[0:-9], None, None, None, None, None]
            final.append(annot)
    
# Writing the required files

with open("C:\\Project_NLP_Final\\Project Dataset\\PreProcessing\\Regex\\CSV_Annotations.csv",'a', encoding = 'utf8', newline='') as outcsv:   
    writer = csv.writer(outcsv, delimiter=',',quotechar = '"')
    for row in final:
        writer.writerow(row)