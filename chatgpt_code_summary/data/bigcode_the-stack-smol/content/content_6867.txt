from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from replacers import *
import pandas as pd
import nltk
import subprocess


def findFreqWord(fuzzyDF):
    f1 = fuzzyDF  # pd.read_csv("SubmittedCSV/fuzzy.csv")
    f2 = pd.DataFrame(columns=['Tweets', 'Classified', 'FreqWord'])
    f3 = pd.read_csv("SubmittedCSV/fuzzyptag.csv", )

    pop_list = list(f3.iloc[:, 0])

    for zero_cl_row in range(f1.__len__()):
        row = 1
        found = False
        splitted_sentence = f1.iloc[zero_cl_row, 0].split()
        print(splitted_sentence)
        for tag in pop_list:
            print("Popular tags:", pop_list)
            for word in splitted_sentence:

                if word in tag and f1.iloc[zero_cl_row, 1] == "Highly Positive":
                    f2 = f2.append(
                        {'Tweets': f1.iloc[zero_cl_row, 0], 'Classified': 'Highly Positive', 'FreqWord': tag},
                        ignore_index=True)
                    found = True
                    row += 1
                elif word in tag and f1.iloc[zero_cl_row, 1] == "Highly Negative":
                    f2 = f2.append(
                        {'Tweets': f1.iloc[zero_cl_row, 0], 'Classified': 'Highly Negative', 'FreqWord': tag},
                        ignore_index=True)
                    found = True
                    row += 1
                elif word in tag and f1.iloc[zero_cl_row, 1] == "Moderately Positive":
                    f2 = f2.append(
                        {'Tweets': f1.iloc[zero_cl_row, 0], 'Classified': 'Moderately Positive', 'FreqWord': tag},
                        ignore_index=True)
                    found = True
                    row += 1
                elif word in tag and f1.iloc[zero_cl_row, 1] == "Moderately Negative":
                    f2 = f2.append(
                        {'Tweets': f1.iloc[zero_cl_row, 0], 'Classified': 'Moderately Negative', 'FreqWord': tag},
                        ignore_index=True)
                    found = True
                    row += 1
                elif word in tag and f1.iloc[zero_cl_row, 1] == "Positive":
                    f2 = f2.append({'Tweets': f1.iloc[zero_cl_row, 0], 'Classified': 'Positive', 'FreqWord': tag},
                                   ignore_index=True)
                    found = True
                    row += 1
                elif word in tag and f1.iloc[zero_cl_row, 1] == "Negative":
                    f2 = f2.append({'Tweets': f1.iloc[zero_cl_row, 0], 'Classified': 'Negative', 'FreqWord': tag},
                                   ignore_index=True)
                    found = True
                    row += 1
                else:
                    print("Unmatched")
            if not found:
                print("NO")
    f2.to_csv("SubmittedCSV/fuzzyfreq.csv", index=False)
    try:
        subprocess.call(['libreoffice','--calc','SubmittedCSV/fuzzyfreq.csv'])
    except OSError:
        print("Works with DEBIAN OS & LIBREOFFICE 5 only \n Use MS Excel or equivalent Software to open : "
              "SubmittedCSV/fuzzyfreq.csv")
    return f2

def pivotTable():
    pass


#       ---------------------------------- SUBMITTED LOGIC - TEST CASE
#       ---------------------------------- #01 UNIT TESTING FAILED ##10, 11, 27, 30
#       ---------------------------------- #02 LOGICAL GLITCH
#       ---------------------------------- #03 COMPLIANCE MISUSE
#       ---------------------------------- #04 MEMDUMP DETECTED
#       ---------------------------------- #05 UNUSED OBJECTS, MEMORY BLOCK 0x0008
# for hosts_row in f1:
#     row = 1
#     found = False
#     # t1=nltk.word_tokenize(hosts_row[0])
#     t1 = hosts_row.split()
#     print("t1=", t1)
#     for master_row in pop_list:
#         print("popular tags=", pop_list)
#         for word in t1:
#
#             if word == master_row[0] and hosts_row[1] == "Highly Positive":
#                 # >>> master_row[0]                                 # Logical glitch, value uncompilable
#                 # 'b'
#                 f2.write(str(hosts_row[1]) + "," + word)            # Will always look for 1st element of string
#                 # >>> hosts_row
#                 # ' neville rooney end ever tons trophy drought httpcocryingeyesjebfkdp,Positive\r\n'
#                 # >>> hosts_row[1]
#                 # 'n'
#                 found = True
#                 row = row + 1
#
#             elif word == master_row[0] and hosts_row[1] == "Highly Negative":
#                 f2.write(str(hosts_row[1]) + "," + str(master_row[0]))
#                 found = True
#                 row = row + 1
#             elif word == master_row[0] and hosts_row[1] == "Moderately Positive":
#                 f2.write(str(hosts_row[1]) + "," + str(master_row[0]))
#                 found = True
#                 row = row + 1
#             elif word == master_row[0] and hosts_row[1] == "Moderately Negative":
#                 f2.write(str(hosts_row[1]) + "," + str(master_row[0]))
#                 found = True
#                 row = row + 1
#             elif word == master_row[0] and hosts_row[1] == "Positive":
#                 f2.write(str(hosts_row[1]) + "," + str(master_row[0]))
#                 # >>> master_row[0]
#                 # 'business'
#                 # >>> hosts_row[1]
#                 # 'n'
#                 found = True
#                 row = row + 1
#             elif word == master_row[0] and hosts_row[1] == "Negative":
#                 f2.write(str(hosts_row[1]) + "," + str(master_row[0]))
#                 found = True
#                 row = row + 1
#
#             # print count
#         if not found:
#             print("no")
#
# print(count)
# f1.close()
# f2.close()
