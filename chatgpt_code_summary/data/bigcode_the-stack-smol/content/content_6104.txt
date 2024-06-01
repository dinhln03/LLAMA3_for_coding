#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/5/2018 1:49 PM 
# @Author : sunyonghai 
# @File : xml_utils.py
# @Software: ZJ_AI
#此程序用于编辑xml文件
# =========================================================
import random
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import os

import cv2

from data_processing.utils.io_utils import *


def read_xml(in_path):
    '''''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    # tree = ET()
    tree = ET.parse(in_path)
    return tree

def write_xml(tree, out_path):
    '''''将xml文件写出
       tree: xml树
       out_path: 写出路径'''
    tree.write(out_path, encoding="utf-8",xml_declaration=True)


def if_match(node, kv_map):
    '''''判断某个节点是否包含所有传入参数属性
       node: 节点
       kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True


# ---------------search -----

def find_nodes(tree, path):
    '''''查找某个路径匹配的所有节点
       tree: xml树
       path: 节点路径'''
    return tree.findall(path)


def get_node_by_keyvalue(nodelist, kv_map):
    '''''根据属性及属性值定位符合的节点，返回节点
       nodelist: 节点列表
       kv_map: 匹配属性及属性值map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes


# ---------------change -----

def change_node_properties(nodelist, kv_map, is_delete=False):
    '''''修改/增加 /删除 节点的属性及属性值
       nodelist: 节点列表
       kv_map:属性及属性值map'''
    for node in nodelist:
        for key in kv_map:
            if is_delete:
                if key in node.attrib:
                    del node.attrib[key]
            else:
                node.set(key, kv_map.get(key))


def change_node_text(nodelist, text, is_add=False, is_delete=False):
    '''''改变/增加/删除一个节点的文本
       nodelist:节点列表
       text : 更新后的文本'''
    for node in nodelist:
        if is_add:
            node.text += text
        elif is_delete:
            node.text = ""
        else:
            node.text = text


def create_node(tag, property_map, content):
    '''''新造一个节点
       tag:节点标签
       property_map:属性及属性值map
       content: 节点闭合标签里的文本内容
       return 新节点'''
    element = Element(tag, property_map)
    element.text = content
    return element


def add_child_node(nodelist, element):
    '''''给一个节点添加子节点
       nodelist: 节点列表
       element: 子节点'''
    for node in nodelist:
        node.append(element)


def del_node_by_tagkeyvalue(nodelist, tag, kv_map):
    '''''同过属性及属性值定位一个节点，并删除之
       nodelist: 父节点列表
       tag:子节点标签
       kv_map: 属性及属性值列表'''
    for parent_node in nodelist:
        children = parent_node.getchildren()
        for child in children:
            if child.tag == tag and if_match(child, kv_map):
                parent_node.remove(child)
#
# if __name__ == "__main__":
#     # 1. 读取xml文件
#     tree = read_xml("./test.xml")
#
#     # 2. 属性修改
#     # A. 找到父节点
#     nodes = find_nodes(tree, "processers/processer")
#     # B. 通过属性准确定位子节点
#     result_nodes = get_node_by_keyvalue(nodes, {"name": "BProcesser"})
#     # C. 修改节点属性
#     change_node_properties(result_nodes, {"age": "1"})
#     # D. 删除节点属性
#     change_node_properties(result_nodes, {"value": ""}, True)
#
#     # 3. 节点修改
#     # A.新建节点
#     a = create_node("person", {"age": "15", "money": "200000"}, "this is the firest content")
#     # B.插入到父节点之下
#     add_child_node(result_nodes, a)
#
#     # 4. 删除节点
#     # 定位父节点
#     del_parent_nodes = find_nodes(tree, "processers/services/service")
#     # 准确定位子节点并删除之
#     target_del_node = del_node_by_tagkeyvalue(del_parent_nodes, "chain", {"sequency": "chain1"})
#
#     # 5. 修改节点文本
#     # 定位节点
#     text_nodes = get_node_by_keyvalue(find_nodes(tree, "processers/services/service/chain"), {"sequency": "chain3"})
#     change_node_text(text_nodes, "new text")
#
#     # 6. 输出到结果文件
#     write_xml(tree, "./out.xml")
#



def modify_label_name(input_path):
    # train VOC2012
    data_paths = [os.path.join(input_path,s) for s in ['train_data']]
    print('Parsing annotation files')

    for data_path in data_paths:
        annot_path = os.path.join(data_path, 'Annotations')
        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
        for annot in annots:
            try:
                et = ET.parse(annot)
                element = et.getroot()

                element_objs = element.findall('object')

                for element_obj in element_objs:
                    node = element_obj.find('name')
                    print(node.text)
                    class_name = element_obj.find('name').text
                    if class_name == 'mn-zgl-pz-cmw-250ml':     # 1
                        node.text = 'mn-zgl-hz-cmw-250ml'

                    # if class_name == 'kkklgz330ml':     # 1
                    #     node.text = 'kkkl-kkkl-gz-yw-330ml'
                    # elif class_name == 'nfsq550ml':     #2
                    #     node.text = 'nfsq-nfsq-pz-yw-550ml'
                    # elif class_name == 'jdbpz500ml':    #3
                    #     node.text = 'jdb-jdb-pz-yw-500ml'
                    # elif class_name == 'wljgz310ml':    #4
                    #     node.text = 'wlj-wlj-gz-yw-310ml'
                    # elif class_name == 'wtnmcgz310ml':  #5
                    #     node.text = 'wt-wtnmc-gz-yw-310ml'
                    # elif class_name == 'ybpz550ml':     #6
                    #     node.text = 'yb-yb-pz-yw-550ml'
                    # elif class_name == 'mdpzqn600ml':   #7
                    #     node.text = 'md-md-pz-qn-600ml'
                    # elif class_name == 'xbgz330ml':     #8
                    #     node.text = 'xb-xb-gz-yw-330ml'
                    # elif class_name == 'fdgz330ml':     #9
                    #     node.text = 'fd-fd-gz-yw-330ml'
                    # elif class_name == 'bsklpz600ml':   #10
                    #     node.text = 'bskl-bskl-pz-yw-600ml'
                    # elif class_name == 'tdyhgz330ml':   #11
                    #     node.text = 'tdyh-tdyh-gz-yw-330ml'
                    # elif class_name == 'qxgz330ml':     #12
                    #     node.text = 'qx-qx-gz-yw-330ml'
                    # elif class_name == 'bwpjgz550ml':   #13
                    #     node.text = 'bw-pj-gz-yw-550ml'
                    # elif class_name == 'qdpjgz330ml':   #14
                    #     node.text = 'qdpj-qdpj-gz-yw-330ml'
                    # elif class_name == 'qoo310ml':      #15
                    #     node.text = 'qoo-qoo-gz-yw-310ml'
                    # elif class_name == 'jtpz560ml':     #16
                    #     node.text = 'jt-jt-pz-yw560ml'
                    # elif class_name == 'mndgz330ml':    #17
                    #     node.text = 'mnd-mnd-gz-yw-330ml'
                    # elif class_name == 'mndgz380ml':    #18
                    #     node.text = 'mnd-mnd-gz-yw-380ml'
                    # elif class_name == 'blcypz550ml':   #19
                    #     node.text = 'blcy-blcy-pz-yw-550ml'
                    # else:
                    #     node.text = 'other'             #20
                    print(node.text)
                write_xml(et, annot)
            except Exception as e:
                print('Exception in pascal_voc_parser: {}'.format(e))
                continue

def modify2_label_name(input_path):
    # train VOC2012
    #dirs = []
    #data_paths = [os.path.join(input_path,s) for s in dirs]
    # data_paths = [os.path.join(input_path,s) for s in ['train_data-2018-3-20_1']]
    print('Parsing annotation files')

    #for data_path in data_paths:
    annot_path = input_path
    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    for annot in annots:
        try:
            et = ET.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')

            for element_obj in element_objs:
                node = element_obj.find('name')
                class_name = element_obj.find('name').text
                if class_name == 'yd-ydmtcqscm-pz-cmw-56g':
                    print(node.text)
                    print(annot)
                    node.text = 'yd-ydwtkxt-pz-qscmw-56g'
                    #print(node.text)
                elif class_name == 'htk-tls-dz-hd-288g':
                    print(node.text)
                    print(annot)
                    node.text = 'htk-tls-dz-hdw-288g'
                    print(node.text)
                elif class_name == 'hwd-hwdfbm-tz-hxw-75g':
                    print(node.text)
                    print(annot)
                    node.text = 'hwd-hwdfbm-tz-hxw-84g'
                    print(node.text)
                elif class_name == 'df-dfqkl-dz-zrqkl--43g':
                    print(node.text)
                    print(annot)
                    node.text = 'df-dfqkl-dz-zrqkl-43g'
                    print(node.text)


                    # elif class_name == 'mn-zgl-pz-cmw-250ml':  # 1
                    #     print(node.text)
                    #     print(annot)
                    #     node.text = 'mn-zgl-hz-cmw-250ml'
                    #     print(node.text)
                    # elif class_name == 'None':  # 1
                    #     print(node.text)
                    #     print(annot)
                    #     node.text = 'yb-ybcjs-pz-yw-555ml'
                    #     print(node.text)
                    # elif class_name == 'db-jdblc-gz-yw-310ml':  # 1
                    #     print(node.text)
                    #     print(annot)
                    #     node.text = 'jdb-jdblc-gz-yw-310ml'
                    #     print(node.text)
                    # elif class_name == 'jdb-jdblc-pz-yw-500ml':  # 1
                    #     print(node.text)
                    #     print(annot)
                    #     node.text = 'jdb-jdb-pz-yw-500ml'
                    #     print(node.text)
                    # elif class_name == 'wlj-wljlc-dz-yw-250ml':  # 1
                    #     print(node.text)
                    #     print(annot)
                        # node.text = 'jdb-jdb-pz-yw-500ml'
                        # print(node.text)
                    # elif class_name == 'mn-zgl-pz-cmw-250ml':     # 1
                    #     node.text = 'mn-zgl-hz-cmw-250ml'
                    #     print(node.text)


                    # elif class_name == 'yl-ylcnn-pz-yw-250ml':    #2
                    #     node.text = 'yl-ylcnn-hz-yw-250ml'
                    # elif class_name == 'lzs-rnbdwhbg-bz-nlw-145g': #3
                    #     node.text = 'lzs-rnbdwhbg-hz-nlw-145g'
                    # elif class_name == 'ksf-ksfbg-bz-qxnmw-125g': #3
                    #     node.text = 'ksf-ksfbg-dz-qxnmw-125g'
                    # elif class_name == 'lfe-lfeyrbttgsq-dz-yrbtr-30g': #4
                    #     node.text = 'lfe-lfeyrbttgsq-hz-yrbtr-30g'
                    # elif class_name == 'df-dfqkl-bz-zrqkl--43g': #5
                    #     node.text = 'df-dfqkl-dz-zrqkl--43g'
                    # elif class_name == 'slj-sljqkl-bz-hsjx-35g': #6
                    #     node.text = 'slj-sljqkl-dz-hsjx-35g'
                    # elif class_name == 'ls-lssp-bz-mgjdyw-70g': #7
                    #     node.text = 'ls-lssp-dz-mgjdyw-70g'
                    # elif class_name == 'wtn-wtnywdn-pz-yw-250ml': #8
                    #     node.text = 'wtn-wtnywdn-hz-yw-250ml'
                    # elif class_name == 'ksf-ksfhsnrm-tz-nr-105g': #9
                    #     node.text = 'ty-tyhsnrm-tz-nr-105g'
                    # elif class_name == 'ty-tyltscnrm-tz-scnr-82.5g': #10
                    #     node.text = 'ksf-ksfltscnrm-tz-scnr-82.5g'
                    # elif class_name == 'yj-pjfz-bz-sjw-100g': #11
                    #     node.text = 'yj-pjfz-dz-sjw-100g'
                    # elif class_name == 'jb-jbjyz-bz-yw-95g': #12
                    #     node.text = 'jb-jbjyz-dz-yw-95g'
                    # elif class_name == 'wwsp-wwxxs-bz-yw-60g': #13
                    #     node.text = 'wwsp-wwxxs-dz-yw-60g'
            write_xml(et, annot)
        except Exception as e:
            print('Exception in pascal_voc_parser: {}'.format(e))
            continue

def get_split():
    str = 'blcy-blcy-pz-yw-550ml'
    class_name = str.split('-')[2]
    print(class_name)

def get_imagenamge_by_label(input_path):
    data_paths = [os.path.join(input_path,s) for s in ['train_data-2018-3-7']]

    for data_path in data_paths:
        annot_path = os.path.join(data_path, 'Annotations')
        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
        for annot in annots:
            try:
                et = ET.parse(annot)
                element = et.getroot()

                element_objs = element.findall('object')

                for element_obj in element_objs:
                    node = element_obj.find('name')
                    class_name = element_obj.find('name').text
                    if class_name == 'qdpj-qdpj-gz-yw-330ml':
                        print(annot)
            except Exception as ex:
                print(ex)





# if __name__ == "__main__":
#     input_path = 'data/'
#     modify_label_name(input_path)

# if __name__ == "__main__":
#     input_path = 'data/train_data-2018-3-7/'
#     rename_image(input_path)

# if __name__ == "__main__":
#     get_split()

# if __name__ == "__main__":
#     input_path = 'data/all_data/'
#     create_Main(input_path)

if __name__ == "__main__":
    input_path = 'D:\\all_data\\predict_data-2018-05-11\\Annotations'
    modify2_label_name(input_path)

# if __name__ == "__main__":
#     input_path = 'data/'
#     get_imagenamge_by_label(input_path)