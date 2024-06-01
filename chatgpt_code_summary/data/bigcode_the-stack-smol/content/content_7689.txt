#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from libs.constants import DEFAULT_ENCODING
from libs.ustr import ustr



XML_EXT = '.xml'
ENCODE_METHOD = DEFAULT_ENCODING

class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def convertPoints2BndBox(self, QPoints):
        points=[(p.x(), p.y()) for p in QPoints]
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'data_set')
        folder.text = self.foldername

        # filename = SubElement(top, 'filename')
        # filename.text = self.filename

        # if self.localImgPath is not None:
        #     localImgPath = SubElement(top, 'path')
        #     localImgPath.text = self.localImgPath

        # source = SubElement(top, 'source')
        # database = SubElement(source, 'database')
        # database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        # segmented = SubElement(top, 'segmented')
        # segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult, parents, children, self_id):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        bndbox['parents'] = parents
        bndbox['children'] = children
        bndbox['self_id'] = self_id
        self.boxlist.append(bndbox)

    def addBehavior(self, label, self_id, start_frame, end_frame, shapes=None):
        bndbox = {}
        bndbox['behavior'] = label
        bndbox['self_id'] = self_id
        bndbox['start_frame'] = start_frame
        bndbox['end_frame'] = end_frame
        bndbox['shapes'] = shapes
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_behavior in self.boxlist:
            object_item = SubElement(top, 'behaviors')
            object_id = SubElement(object_item, 'behavior_id')
            object_id.text = str(each_behavior['self_id'])
            name = SubElement(object_item, 'behavior')
            name.text = str(each_behavior['behavior'])
            start = SubElement(object_item, 'start_frame')
            start.text = str(each_behavior['start_frame'])
            if start.text == "":
                start.text = "undefined"
            end = SubElement(object_item, 'end_frame')
            end.text = str(each_behavior['end_frame'])
            if end.text == "":
                end.text = "undefined"

            if each_behavior['shapes'] != None:
                shapes = SubElement(object_item, 'bounding_boxes')
                for each_shape in each_behavior['shapes']:
                    bounding_box = self.convertPoints2BndBox(each_shape.points)

                    shape = SubElement(shapes, 'bounding_box')
                    frame = SubElement(shape, 'frame')
                    frame.text = str(each_shape.filename)

                    bndbox = SubElement(shape, 'bndbox')
                    xmin = SubElement(bndbox, 'xmin')
                    xmin.text = str(bounding_box[0])
                    ymin = SubElement(bndbox, 'ymin')
                    ymin.text = str(bounding_box[1])
                    xmax = SubElement(bndbox, 'xmax')
                    xmax.text = str(bounding_box[2])
                    ymax = SubElement(bndbox, 'ymax')
                    ymax.text = str(bounding_box[3])

        # all_ids = []
        # for each_object in self.boxlist:
        #     all_ids.append(each_object['self_id'])

        # for each_object in self.boxlist:
        #     object_item = SubElement(top, 'object')

        #     object_id = SubElement(object_item, 'object_id')
        #     object_id.text = str(each_object['self_id'])

        #     name = SubElement(object_item, 'name')
        #     real_name = ustr(each_object['name'])

        #     name.text = str()
        #     for letter in real_name:
        #         if letter != ' ':
        #             name.text += letter
        #         else:
        #             name.text = str()

        #     if len(each_object['parents']) != 0:
        #         parents = SubElement(object_item, 'has_parents')
        #         for each_id in each_object['parents']:
        #             if each_id in all_ids:
        #                 parent = SubElement(parents, 'parent')
        #                 parent.text = str(each_id)


        #     if len(each_object['children']) != 0:
        #         children = SubElement(object_item, 'has_children')
        #         for each_id in each_object['children']:
        #             if each_id in all_ids:
        #                 child = SubElement(children, 'child')
        #                 child.text = str(each_id)

        #     pose = SubElement(object_item, 'pose')
        #     pose.text = "Unspecified"
        #     truncated = SubElement(object_item, 'truncated')
        #     if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin']))== 1):
        #         truncated.text = "1" # max == height or min
        #     elif (int(float(each_object['xmax']))==int(float(self.imgSize[1]))) or (int(float(each_object['xmin']))== 1):
        #         truncated.text = "1" # max == width or min
        #     else:
        #         truncated.text = "0"
        #     difficult = SubElement(object_item, 'difficult')
        #     difficult.text = str( bool(each_object['difficult']) & 1 )
        #     bndbox = SubElement(object_item, 'bndbox')
        #     xmin = SubElement(bndbox, 'xmin')
        #     xmin.text = str(each_object['xmin'])
        #     ymin = SubElement(bndbox, 'ymin')
        #     ymin.text = str(each_object['ymin'])
        #     xmax = SubElement(bndbox, 'xmax')
        #     xmax.text = str(each_object['xmax'])
        #     ymax = SubElement(bndbox, 'ymax')
        #     ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.behaviors = []
        self.verified = False
        try:
            self.readBehavior()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def getBehaviors(self):
        return self.behaviors

    def addBehavior(self, behavior_name, behavior_id, starting_frame, ending_frame, shapes=None):
        self.behaviors.append((behavior_name, behavior_id, starting_frame, ending_frame, shapes))

    def addShape(self, label, bndbox, difficult, parents, children, object_id):

        parent_ids = []
        try:
            for item in parents.findall('parent'):
                parent_ids.append(int(item.text))
        except:
            pass
        
        child_ids = []
        try:
            for item in children.findall('child'):
                child_ids.append(int(item.text))
        except:
            pass

        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, parent_ids, child_ids, object_id, None, None, difficult, ))

    def readBehavior(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        for object_iter in xmltree.findall('behaviors'):
            label = object_iter.find('behavior').text
            object_id = int(object_iter.find('behavior_id').text)
            start_frame = object_iter.find('start_frame').text
            end_frame = object_iter.find('end_frame').text
            shapes = object_iter.find('bounding_boxes')

            bounding_boxes = []
            for shape_tier in shapes.findall('bounding_box'):
                box = {}
                bndbox = shape_tier.find("bndbox")
                box["bndbox"] = bndbox
                frame = shape_tier.find("frame")
                box["frame"] = frame.text
                bounding_boxes.append(box)

            self.addBehavior(label, object_id, start_frame, end_frame, bounding_boxes)
            
        return True

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            parents = object_iter.find('has_parents')
            children = object_iter.find('has_children')
            object_id = int(object_iter.find('object_id').text)
            # Add chris
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.addShape(label, bndbox, difficult, parents, children, object_id)
        return True
