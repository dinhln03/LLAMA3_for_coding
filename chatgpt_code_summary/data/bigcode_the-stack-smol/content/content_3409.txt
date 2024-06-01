#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets, QtGui, QtCore
import sys, os.path as op
path1 = op.join( op.abspath(op.dirname(__file__)), '..', 'Structure')
path2 = op.join( op.abspath(op.dirname(__file__)), '..')
sys.path.append(path1)
sys.path.append(path2)
from Structure import *
from VisObject import *

class SubVision( QtWidgets.QWidget ):
    """ Базовый класс-окно для показа подчиненных объектов """
    def __init__( self, main_object, is_change=True, parent=None ):
        super().__init__( parent=parent )
        #Устанавливаем главный объект
        self.__obj = main_object
        #Устанавливаем параметр возможности изменения элементов (по умолчанию - Да)
        self.is_change = is_change
        self.initUI()
    
    def initUI( self ):
        ''' Инициализируем содержимое окна '''
        
        #Добавляем окно данных и устанавливаем в него подчиненные объекты
        self.sub_objs = QtWidgets.QListWidget( )
        for obj in self.__obj.sub_objects:
            #Делаем ячейку
            a = QtWidgets.QListWidgetItem()
            #Устанавливаем в ней подчиненный базовому объект
            a.sub_obj = obj
            #Устанавливаем в ней текст-имя объекта подчиненного объекта
            a.setText( obj.name )
            #Добавляем в список
            self.sub_objs.addItem( a )
            
        #Объявляем форму и добавляем в нее список подчиненных объектов
        self.form = QtWidgets.QFormLayout()
        self.form.addRow(self.sub_objs)
        
        self.setLayout(self.form)
        #Соединяем двойной щелчок с методом
        self.sub_objs.itemDoubleClicked.connect( self.isDoubleClicked )
    
    def isDoubleClicked( self, obj ):
        #Если окно возможно изменить, вызываем окно изменения, иначе - окно просмотра
        if self.is_change:
            sub_window = ChangeVisObject( obj.sub_obj, parent=self )
        else:
            sub_window = SimpleVisObject( obj.sub_obj, parent=self )
        sub_window.setWindowTitle( "Редактирование объекта: " + obj.sub_obj.name )
        #Делаем это или родительское окно неактивным
        if self.parent() is None:
            self.setEnabled( False )
        else:
            self.parent().setEnabled( False )
        #Делаем дочернее окно активным и показываем его
        sub_window.setEnabled( True )
        sub_window.show()
