# import pandas as pd
#
# csv_data = pd.read_csv('E:\\AI_Object_Detect\\Code\\TextRecognitionDataGenerator\\idcard_file.txt', sep=',', header=0, encoding='UTF-8')
# N = 5
# csv_batch_data = csv_data.tail(N)
# print(csv_batch_data.shape)

import csv
import os
idcard_file = 'E:\\AI_Object_Detect\\Code\\TextRecognitionDataGenerator\\idcard_file.txt'
idcard_data = []
with open(idcard_file, 'r', encoding='UTF-8') as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    birth_header = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        tmp_str = row[10]
        if 'issueAuthority' in tmp_str:
            front = row[10].split(':')[1] + row[11].split(':')[1]
            idcard_data.append(front.replace('"', '').replace("}",''))
        elif 'address' in tmp_str:
            back = row[10].split(':')[1] + row[11].split(':')[1] + row[12].split(':')[1] + row[13].split(':')[1] + row[14].split(':')[1] + row[15].split(':')[1]
            idcard_data.append(back.replace('"', '').replace("}",''))

        # print(str + '\r\n')

lang = 'char_std_5991'
with open(os.path.join('dicts', lang + '.txt'), 'r', encoding="utf8", errors='ignore') as d:
    lang_dict = d.readlines()
    lang_dict = [ch.strip('\n') for ch in lang_dict]


for text in idcard_data:
    for character in text:
        try:
            p = lang_dict.index(character)
        except ValueError:
            lang_dict.append(character)
            print(character)



# file=open('texts/data.txt','w+', encoding='UTF-8')
# for  strgin in idcard_data:
#     file.write(strgin + '\n')
# file.close()

# for cnt in idcard_data:
#     print(cnt)
#     print('\n')



# idcard_data = [[float(x) for x in row] for row in idcard_data]  # 将数据从string形式转换为float形式


# birth_data = np.array(birth_data)  # 将list数组转化成array数组便于查看数据结构
# birth_header = np.array(birth_header)
# print(birth_data.shape)  # 利用.shape查看结构。
# print(birth_header.shape)